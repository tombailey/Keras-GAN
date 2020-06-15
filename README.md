## Minecraft Skin Generator
This project uses Artificial Intelligence (AI) to "dream up" Minecraft skins. It uses a Deep Convolutional Generative Adversarial Network (DCGAN) trained on existing Minecraft skins to "understand" what they look like and then create its own.

## Installation
Please install Python 3.7.4 before continuing.

    $ git clone https://github.com/tombailey/Keras-GAN.git
    $ pip3 install -r requirements.txt

## Training
Please gather some existing Minecraft skins for training. Place them as PNG files in ```./images```.

    $ python3 main.py

Note, if you are able to, you might want to use the GPU variant of TensorFlow to speed training up.

During training, you might want to periodically check ```./images/generated``` to see how the model is performing.

## Generating skins
Once the model has been trained the weights for the generator will be saved to ```./saved_model/generator.h5``` so they can be used to generate skins. You can run the same script which will detect and use the saved weights to generate skins.

    $ python3 main.py

Check for the generated skin at ```./images/generated/example.png```.

## Credit   

The DCGAN implementation is based on eriklindernoren's implementation of a DCGAN. This repository is a fork of [eriklindernoren's Keras-GAN repository](https://github.com/eriklindernoren/Keras-GAN), in which you can find the original implementation.

eriklindernoren's implementation is based on the following [paper](https://arxiv.org/abs/1511.06434).
