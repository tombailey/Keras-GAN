## Minecraft Skin Generator
This project uses Artificial Intelligence (AI) to "dream up" Minecraft skins. It is trained on existing Minecraft skins to "understand" what they look like so it can create its own.

## New model

The main branch has been updated to use a diffusion based model instead of a wgan model. The new model results in higher quality skins but requires substantially more resources. A GPU with enough VRAM for your dataset is recommended. If you don't have access to hardware like that, you can still use the older wgan model with the [wgan branch](https://github.com/tombailey/Minecraft-Skin-Generator/tree/wgan).

## Installation
Please install Python 3.9 before continuing.

    $ git clone https://github.com/tombailey/Minecraft-Skin-Generator.git
    $ pip3 install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu117

## Training
Please gather some existing Minecraft skins for training. Place them as PNG files in `./data`.

    $ python3 main.py --mode train --dataset data --epochs 10

You might want to use `trim.py` and `deduplicate.py` to clean the data before training.

During training, you might want to periodically check `./output/samples` to see how the model is performing.

## Generating skins
Once the model has been trained it will be saved to `./output/unet` so it can be used to generate skins. You can run the same script to generate skins.

    $ python3 main.py --mode generate

Check for the generated skins at `./output/generated-*.png`.

## Credit   

The diffusion model used to generate images is based heavily on [Huggingface's DDPM notebook](https://github.com/huggingface/notebooks/blob/8392fcc5d61396e39159c9f582c131599a48579b/diffusers/training_example.ipynb).
