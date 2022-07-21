from os import mkdir
from os.path import exists

import numpy as np

import torch
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from ml.pytorch.image_dataset import ImageDataset
from ml.pytorch.wgan_gp.discriminator import Discriminator
from ml.pytorch.wgan_gp.generator import Generator


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

    def compute_gradient_penalty(self, discriminator, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        Tensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        # Random weight term for interpolation between real and fake samples
        alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = discriminator(interpolates)
        fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
        # Get gradient w.r.t. interpolates
        gradients = grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def train(
        self,
        image_dataset: ImageDataset,
        learning_rate: float = 0.0002,
        betas: (float, float) = (0.5, 0.999),
        lambda_gradient_penalty: float = 10,
        batch_size: int = 64,
        workers: int = 8,
        epochs: int = 100,
        discriminator_steps: int = 5,
        sample_interval: int = 1000,
        sample_folder: str = 'samples',
        generator_save_file: str = 'generator.model',
        discriminator_save_file: str = 'discriminator.model'
    ):
        if not exists(sample_folder):
            mkdir(sample_folder)

        generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=learning_rate, betas=(betas[0], betas[1]))
        discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=learning_rate, betas=(betas[0], betas[1]))

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

                # Configure input
                real_imgs = Variable(imgs.type(Tensor))

                # ---------------------
                #  Train Discriminator
                # ---------------------

                discriminator_optimizer.zero_grad()

                # Sample noise as generator input
                z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], self.latent_space_dimension))))

                # Generate a batch of images
                fake_imgs = self.generator(z)

                # Real images
                real_validity = self.discriminator(real_imgs)
                # Fake images
                fake_validity = self.discriminator(fake_imgs)
                # Gradient penalty
                gradient_penalty = self.compute_gradient_penalty(self.discriminator, real_imgs.data, fake_imgs.data)
                # Adversarial loss
                discriminator_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gradient_penalty * gradient_penalty

                discriminator_loss.backward()
                discriminator_optimizer.step()

                generator_optimizer.zero_grad()

                # Train the generator every discriminator_steps
                if i % discriminator_steps == 0:

                    # -----------------
                    #  Train Generator
                    # -----------------

                    # Generate a batch of images
                    fake_imgs = self.generator(z)
                    # Loss measures generator's ability to fool the discriminator
                    # Train on fake images
                    fake_validity = self.discriminator(fake_imgs)
                    generator_loss = -torch.mean(fake_validity)

                    generator_loss.backward()
                    generator_optimizer.step()

                    print(
                        f'[Epoch {epoch}/{epochs}] [Batch {i}/{len(data_loader)}] ' +
                        f'[D loss: {discriminator_loss.item()}] [G loss: {generator_loss.item()}]'
                    )

                    if batches_done % sample_interval == 0:
                        save_image(fake_imgs.data[:25], f'{sample_folder}/{batches_done}.png', nrow=5, normalize=True)

                    batches_done += discriminator_steps
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
        save_image(gen_imgs.data[:25], f'{sample_folder}/generated.png', nrow=5, normalize=True)
