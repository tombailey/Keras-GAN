from os import mkdir
from os.path import exists

import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from ml.pytorch.image_dataset import ImageDataset
from ml.pytorch.wgan.discriminator import Discriminator
from ml.pytorch.wgan.generator import Generator


class ImageWgan:
    def __init__(
        self,
        image_shape: (int, int, int),
        latent_space_dimension: int = 100,
        use_cuda: bool = False,
        generator_saved_model: str or None = None,
        discriminator_saved_model: str or None = None
    ):
        self.generator = Generator(image_shape, latent_space_dimension, use_cuda, generator_saved_model)
        self.discriminator = Discriminator(image_shape, use_cuda, discriminator_saved_model)

        self.image_shape = image_shape
        self.latent_space_dimension = latent_space_dimension
        self.use_cuda = use_cuda
        if use_cuda:
            self.generator.cuda()
            self.discriminator.cuda()

    def train(
        self,
        image_dataset: ImageDataset,
        learning_rate: float = 0.00005,
        batch_size: int = 64,
        workers: int = 8,
        epochs: int = 100,
        clip_value: float = 0.01,
        discriminator_steps: int = 5,
        sample_interval: int = 1000,
        sample_folder: str = 'samples',
        generator_save_file: str = 'generator.model',
        discriminator_save_file: str = 'discriminator.model'
    ):
        if not exists(sample_folder):
            mkdir(sample_folder)

        generator_optimizer = torch.optim.RMSprop(self.generator.parameters(), lr=learning_rate)
        discriminator_optimizer = torch.optim.RMSprop(self.discriminator.parameters(), lr=learning_rate)

        Tensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor

        data_loader = torch.utils.data.DataLoader(
            image_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers
        )
        batches_done = 0
        for epoch in range(epochs):
            for i, imgs in enumerate(data_loader):
                real_imgs = Variable(imgs.type(Tensor))

                discriminator_optimizer.zero_grad()

                # Sample noise as generator input
                z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], self.latent_space_dimension))))

                fake_imgs = self.generator(z).detach()
                # Adversarial loss
                discriminator_loss = -torch.mean(self.discriminator(real_imgs)) + torch.mean(self.discriminator(fake_imgs))

                discriminator_loss.backward()
                discriminator_optimizer.step()

                # Clip weights of discriminator
                for p in self.discriminator.parameters():
                    p.data.clamp_(-clip_value, clip_value)

                # Train the generator every n_critic iterations
                if i % discriminator_steps == 0:
                    generator_optimizer.zero_grad()

                    # Generate a batch of images
                    gen_imgs = self.generator(z)
                    # Adversarial loss
                    generator_loss = -torch.mean(self.discriminator(gen_imgs))

                    generator_loss.backward()
                    generator_optimizer.step()

                    print(
                        f'[Epoch {epoch}/{epochs}] [Batch {batches_done % len(data_loader)}/{len(data_loader)}] ' +
                        f'[D loss: {discriminator_loss.item()}] [G loss: {generator_loss.item()}]'
                    )

                if batches_done % sample_interval == 0:
                    save_image(gen_imgs.data[:25], f'{sample_folder}/{batches_done}.png', nrow=5, normalize=True)
                batches_done += 1
            self.discriminator.save(discriminator_save_file)
            self.generator.save(generator_save_file)

    def generate(
        self,
        sample_folder: str = 'samples'
    ):
        if not exists(sample_folder):
            mkdir(sample_folder)

        Tensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        z = Variable(Tensor(np.random.normal(0, 1, (self.image_shape[0], self.latent_space_dimension))))
        gen_imgs = self.generator(z)
        generator_loss = -torch.mean(self.discriminator(gen_imgs))
        generator_loss.backward()
        save_image(gen_imgs.data[:25], f'{sample_folder}/generated.png', nrow=5, normalize=True)
