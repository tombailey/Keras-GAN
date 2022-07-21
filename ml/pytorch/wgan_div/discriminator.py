import numpy as np

import torch.nn as nn
import torch


class Discriminator(nn.Module):
    def __init__(
        self,
        image_shape: (int, int, int),
        use_cuda: bool = False,
        saved_model: str or None = None
    ):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(image_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )
        if saved_model is not None:
            self.model.load_state_dict(
                torch.load(
                    saved_model,
                    map_location=torch.device('cuda' if use_cuda else 'cpu')
                )
            )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity

    def save(self, to):
        torch.save(self.model.state_dict(), to)
