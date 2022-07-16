import argparse
from os.path import exists

from ml.pytorch.image_dataset import ImageDataset
from ml.pytorch.wgan.image_wgan import ImageWgan


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=50, help='number of epochs for training')
parser.add_argument('--dataset', type=str, default='data', help='a folder containing the image dataset')
parser.add_argument('--mode', type=str, default='generate', help='"train" or "generate"')
parser.add_argument('--samples', type=str, default='samples', help='a folder to store samples')
parser.add_argument('--discriminator', type=str, default='discriminator.model', help='a file for loading/saving the discriminator')
parser.add_argument('--generator', type=str, default='generator.model', help='a file for loading/saving the generator')
opt = parser.parse_args()


def init():
    dataset_folder = opt.dataset
    generated_samples_folder = opt.samples
    discriminator_saved_model = opt.discriminator
    generator_saved_model = opt.generator

    mode = opt.mode
    image_wgan = ImageWgan(
        image_shape=(4, 64, 64),
        latent_space_dimension=100,
        use_cuda=True,
        generator_saved_model=generator_saved_model if exists(generator_saved_model) else None,
        discriminator_saved_model=discriminator_saved_model if exists(discriminator_saved_model) else None
    )
    if mode == 'train':
        image_wgan.train(
            epochs=opt.epochs,
            image_dataset=ImageDataset(dataset_folder),
            sample_folder=generated_samples_folder,
            generator_save_file=generator_saved_model,
            discriminator_save_file=discriminator_saved_model
        )
    elif mode == 'generate':
        image_wgan.generate(
            sample_folder=generated_samples_folder
        )
    else:
        raise ValueError('Mode {mode} not recognized')


if __name__ == '__main__':
    init()
