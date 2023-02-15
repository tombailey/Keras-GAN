from os import listdir
from os.path import exists, isfile, join
from typing import List

from PIL import Image
import torch
from denoising_diffusion_pytorch import GaussianDiffusion, Unet, Trainer


def determine_last_milestone(milestones_folder: str) -> int or None:
    milestones = [
        int(milestone_file.split('.pt')[0].split('model-')[-1], 10)
        for milestone_file in listdir(milestones_folder)
        if isfile(join(milestones_folder, milestone_file)) and milestone_file.endswith('.pt')
    ]
    return max(milestones, default=None)


class ImageDiffusion:
    def __init__(
        self,
        image_shape: (int, int, int) = (64, 64, 4),
        saved_output_folder: str = 'output',
        use_cuda: bool = torch.cuda.is_available()
    ):
        if not image_shape[0] == image_shape[1]:
            raise ValueError('Width and height of image shape should match.')

        self.image_size = image_shape[0]
        self.use_cuda = use_cuda

        self.saved_output_folder = saved_output_folder

        model = Unet(
            dim=64,
            dim_mults=(1, 2, 4, 8),
            channels=image_shape[2]
        )
        if use_cuda:
            model = model.cuda()

        self.diffusion = GaussianDiffusion(
            model,
            image_size=image_shape[0],
            timesteps=1000,
            sampling_timesteps=250,
            loss_type='l1'
        )
        if use_cuda:
            self.diffusion = self.diffusion.cuda()

        if exists(saved_output_folder):
            last_milestone = determine_last_milestone(saved_output_folder)
            if last_milestone is not None:
                model_path = join(saved_output_folder, f'model-{last_milestone}.pt')
                data = torch.load(model_path)
                self.diffusion.load_state_dict(data['model'])
                print(f'Loaded {model_path}')

    def train(
        self,
        dataset_folder: str,
        training_steps: int = 200000,
        training_batch_size: int = 64
    ):
        Trainer(
            self.diffusion,
            dataset_folder,
            train_batch_size=training_batch_size,
            train_lr=8e-5,
            train_num_steps=training_steps,
            gradient_accumulate_every=2,
            ema_decay=0.995,
            amp=False,
            augment_horizontal_flip=False,
            results_folder=self.saved_output_folder
        ).train()

    def generate(
        self,
        size=1
    ) -> List[Image.Image]:
        tensors = [
            tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8)
            for tensor in self.diffusion.sample(batch_size=size)
        ]
        return [
            Image.fromarray(tensor.numpy(), mode='RGBA')
            for tensor in tensors
        ]
