import torch.nn as nn


class BasalMLP(nn.Module):
    """
    MLP for learning a latent basal cell state representation.

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
        super(BasalMLP, self).__init__()

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


    def forward(self, x):
        x = self.encoder_z(x)
        intrinsic_mean = self.intrinsic_mean_layer(x)

        return intrinsic_mean, 0, 0


