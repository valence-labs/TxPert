import torch
import torch.nn as nn


class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) for learning a latent basal cell state representation.

    Args:
        input_dim (int): Dimension of the input data.
        hidden_dim (int): Dimension of the hidden layers.
        latent_dim (int): Dimension of the latent space.
        dropout (float): Dropout rate.
        use_batch_norm (bool): Whether to use batch normalization.
        use_layer_norm (bool): Whether to use layer normalization.
        decode_latent (bool): Whether to decode the latent space.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        latent_dim: int = 64,
        dropout: float = 0.2,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        decode_latent: bool = False,
    ):
        super(VAE, self).__init__()

        self.no_plot = False
        self.latent_dim = latent_dim

        # Encoder
        encoder_layers = [
            nn.Linear(input_dim, hidden_dim),
            (
                nn.BatchNorm1d(hidden_dim)
                if use_batch_norm
                else nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
            ),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        ]
        self.encoder_z = nn.Sequential(*encoder_layers)

        self.intrinsic_mean_layer = nn.Linear(hidden_dim, latent_dim)
        self.intrinsic_log_var_layer = nn.Linear(hidden_dim, latent_dim)

        self.decoder_z = nn.Linear(latent_dim, latent_dim) if decode_latent else None

    def forward(self, x):
        x = self.encoder_z(x)
        intrinsic_mean, intrinsic_log_var = self.intrinsic_mean_layer(
            x
        ), self.intrinsic_log_var_layer(x)

        z_intrinsic = self.reparameterize(intrinsic_mean, intrinsic_log_var)

        if self.decoder_z is not None:
            z_intrinsic = self.decoder_z(z_intrinsic)

        return z_intrinsic, intrinsic_mean, intrinsic_log_var

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std
