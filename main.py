import argparse
from os import path

from ml.pytorch.diffusion.image_diffusion import ImageDiffusion

parser = argparse.ArgumentParser()
parser.add_argument('--training_steps', type=int, default=200000, help='number of training steps')
parser.add_argument('--dataset', type=str, default='data', help='a folder containing the image dataset')
parser.add_argument('--mode', type=str, default='generate', help='"train" or "generate"')
parser.add_argument('--output', type=str, default='output', help='a folder to save the outputs (logs, model, samples)')
opt = parser.parse_args()


def init():
    mode = opt.mode
    model = ImageDiffusion(
        saved_output_folder=opt.output
    )
    if mode == 'train':
        model.train(
            dataset_folder=opt.dataset,
            training_steps=opt.training_steps,
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
