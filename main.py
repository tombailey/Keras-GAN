import argparse
from os import path

from ml.pytorch.diffusion.image_diffusion import ImageDiffusion

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=20, help='number of epochs for training')
parser.add_argument('--dataset', type=str, default='data', help='a folder containing the image dataset')
parser.add_argument('--mode', type=str, default='generate', help='"train" or "generate"')
parser.add_argument('--output', type=str, default='output', help='a folder to save the outputs (logs, model, samples)')
opt = parser.parse_args()


def init():
    dataset_folder = opt.dataset

    mode = opt.mode
    model = ImageDiffusion(
        saved_output_folder=path.join(opt.output, 'unet')
    )
    if mode == 'train':
        model.train(
            dataset_folder=dataset_folder,
            epochs=opt.epochs,
            output_folder=opt.output
        )
    elif mode == 'generate':
        [
            generated.save(path.join(opt.output, f'generated-{index}.png'))
            for index, generated in enumerate(model.generate(10))
        ]
    else:
        raise ValueError(F'Mode {mode} not recognized')


if __name__ == '__main__':
    init()
