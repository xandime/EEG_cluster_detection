import torch
import torch.nn as nn


class AutoencoderStandard(nn.Module):
    """Fully-connected autoencoder for tabular EEG data.

    Architecture:
      Encoder: Linear(input_dim -> 128) + ReLU,
               Linear(128 -> 64) + ReLU,
               Linear(64 -> bottleneck_dim)
      Decoder: Linear(bottleneck_dim -> 64) + ReLU,
               Linear(64 -> 128) + ReLU,
               Linear(128 -> input_dim)

    forward(x) -> (x_recon, z)
    encode(x) -> z
    """

    def __init__(self, input_dim: int, bottleneck_dim: int):
        super().__init__()
        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, bottleneck_dim),
        )
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, input_dim),
        )

    def forward(self, x: torch.Tensor):
        """Pass input through encoder and decoder and return reconstruction and latent.

        Args:
            x: tensor of shape (batch, input_dim)
        Returns:
            x_recon: tensor of shape (batch, input_dim)
            z: tensor of shape (batch, bottleneck_dim)
        """
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

    def encode(self, x: torch.Tensor):
        """Return only the latent representation z for input x."""
        return self.encoder(x)



