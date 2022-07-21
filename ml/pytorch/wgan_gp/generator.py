import numpy as np

import torch.nn as nn
import torch


class Generator(nn.Module):
    def __init__(
        self,
        image_shape: (int, int, int),
        latent_space_dimension: int,
        use_cuda: bool = False,
        saved_model: str or None = None
    ):
        super(Generator, self).__init__()

        self.image_shape = image_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_space_dimension, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(image_shape))),
            nn.Tanh()
        )
        if saved_model is not None:
            self.model.load_state_dict(
                torch.load(
                    saved_model,
                    map_location=torch.device('cuda' if use_cuda else 'cpu')
                )
            )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.image_shape)
        return img

    def save(self, to):
        torch.save(self.model.state_dict(), to)
