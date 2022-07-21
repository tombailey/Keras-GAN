## Minecraft Skin Generator
This project uses Artificial Intelligence (AI) to "dream up" Minecraft skins. It is trained on existing Minecraft skins to "understand" what they look like so it can create its own.

## Installation
Please install Python 3.8 before continuing.

    $ git clone https://github.com/tombailey/Minecraft-Skin-Generator.git
    $ pip3 install -r requirements.txt -f https://download.pytorch.org/whl/cu113/torch_stable.html

## Training
Please gather some existing Minecraft skins for training. Place them as PNG files in `./data`.

    $ python3 main.py --mode train --model wgandiv --dataset data

You might want to use `trim.py` and `deduplicate.py` to clean the data before training.

During training, you might want to periodically check `./samples` to see how the model is performing.

If you want to visualize the loss, save stdout to a log file:

    $ python3 main.py --mode train --model wgandiv --dataset data > stdout.log

Then parse the log file and create a csv file with the loss data:

    $ python3 parse_loss.py stdout.log loss.csv

Finally, visualize the loss over time:

    $ python3 visualize_loss.py loss.csv

## Generating skins
Once the models have been trained they will be saved to `./discriminator.model` and `./generator.model` so they can be used to generate skins. You can run the same script to generate skins.

    $ python3 main.py --mode generate --model wgandiv --samples results

Check for the generated skins at `./results/*.png`.

## Credit   

The GAN used to generate images is based heavily on [eriklindernoren's WGAN implementation](https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan/wgan.py).
