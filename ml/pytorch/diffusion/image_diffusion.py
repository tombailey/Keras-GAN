from os import path, makedirs
from typing import List

from accelerate import Accelerator, notebook_launcher

from tqdm.auto import tqdm

from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup

import torch
from torch.nn import functional
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets import load_dataset

from PIL import Image


def transformer(preprocess):
    def transform(batch):
        batch["image"] = [preprocess(image.convert("RGBA")) for image in batch["image"]]
        return batch

    return transform


def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGBA', size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


def evaluate(
    evaluating_batch_size: int,
    epoch: int,
    seed: int,
    output_folder: str,
    pipeline: DDPMPipeline
):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size=evaluating_batch_size,
        generator=torch.manual_seed(seed),
    )["sample"]

    # Make a grid out of the images
    image_grid = make_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = path.join(output_folder, "samples")
    makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")


class ImageDiffusion:
    def __init__(
        self,
        image_shape: (int, int, int) = (64, 64, 4),
        saved_output_folder: str or None = None
    ):
        if not image_shape[0] == image_shape[1]:
            raise ValueError("Width and height of image shape should match.")

        self.image_size = image_shape[0]

        if path.exists(saved_output_folder):
            self.model = UNet2DModel.from_pretrained(saved_output_folder)
        else:
            self.model = UNet2DModel(
                sample_size=image_shape[0],  # the target image resolution
                in_channels=image_shape[2],  # the number of input channels, 3 for RGB images
                out_channels=image_shape[2],  # the number of output channels
                layers_per_block=2,  # how many ResNet layers to use per UNet block
                block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channes for each UNet block
                down_block_types=(
                    "DownBlock2D",  # a regular ResNet downsampling block
                    "DownBlock2D",
                    "DownBlock2D",
                    "DownBlock2D",
                    "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                    "DownBlock2D",
                ),
                up_block_types=(
                    "UpBlock2D",  # a regular ResNet upsampling block
                    "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D"
                )
            )
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000, tensor_format="pt")

    def train(
        self,
        dataset_folder: str,
        epochs: int = 20,
        training_batch_size: int = 64,
        evaluating_batch_size: int = 64,
        learning_rate: float = 1e-4,
        lr_warmup_steps: int = 500,
        output_folder: str = 'output',
        save_model_interval=1,
        save_image_interval=1
    ):
        dataset = load_dataset(dataset_folder, data_files={"train": f'*.png'})["train"]

        preprocess = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        dataset.set_transform(transformer(preprocess))
        train_dataloader = DataLoader(dataset, batch_size=training_batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * epochs),
        )

        args = (
            self.model,
            epochs,
            self.noise_scheduler,
            optimizer,
            train_dataloader,
            lr_scheduler,
            evaluating_batch_size,
            output_folder,
            save_model_interval,
            save_image_interval
        )
        notebook_launcher(self.training_loop, args, num_processes=1)

    def training_loop(
        self,
        model,
        epochs: int,
        noise_scheduler,
        optimizer,
        train_dataloader,
        lr_scheduler,
        evaluating_batch_size: int,
        output_folder: str = 'output',
        save_model_interval=1,
        save_image_interval=1
    ):
        # Initialize accelerator and tensorboard logging
        accelerator = Accelerator(
            mixed_precision='fp16',
            gradient_accumulation_steps=1,
            log_with="tensorboard",
            logging_dir=path.join(output_folder, "logs")
        )
        if accelerator.is_main_process:
            accelerator.init_trackers("train_example")

        # Prepare everything
        # There is no specific order to remember, you just need to unpack the
        # objects in the same order you gave them to the prepare method.
        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, lr_scheduler
        )

        global_step = 0

        # Now you train the model
        for epoch in range(epochs):
            progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")

            for step, batch in enumerate(train_dataloader):
                clean_images = batch['image']
                # Sample noise to add to the images
                noise = torch.randn(clean_images.shape).to(clean_images.device)
                bs = clean_images.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (bs,),
                                          device=clean_images.device).long()

                # Add noise to the clean images according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_images = self.noise_scheduler.add_noise(clean_images, noise, timesteps)

                with accelerator.accumulate(model):
                    # Predict the noise residual
                    noise_pred = model(noisy_images, timesteps)["sample"]
                    loss = functional.mse_loss(noise_pred, noise)
                    accelerator.backward(loss)

                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                progress_bar.update(1)
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                global_step += 1

            # After each epoch you optionally sample some demo images with evaluate() and save the model
            if accelerator.is_main_process:
                pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

                if (epoch + 1) % save_image_interval == 0 or epoch == epochs - 1:
                    seed = 0
                    evaluate(evaluating_batch_size, epoch, seed, output_folder, pipeline)

                if (epoch + 1) % save_model_interval == 0 or epoch == epochs - 1:
                    pipeline.save_pretrained(output_folder)

    def generate(
        self,
        size=1
    ) -> List[Image.Image]:
        pipeline = DDPMPipeline(unet=self.model, scheduler=self.noise_scheduler)
        return pipeline(
            batch_size=size,
            generator=torch.manual_seed(torch.seed()),
        )["sample"]
