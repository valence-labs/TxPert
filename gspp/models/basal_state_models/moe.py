import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_

class MixtureLinear(nn.Module):
    """
    Mixture of linear layers.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        self.weight = nn.Parameter(torch.empty((out_features, in_features, rank)))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, rank))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = (0.02 * self.rank**-0.5) ** 0.5
        trunc_normal_(self.weight, std=std)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor, coef: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: Tensor of shape (B, in_features)
            coef: Either a global coefficient of shape (1, rank)
                  or per-sample coefficients of shape (B, rank)
        Returns:
            Tensor of shape (B, out_features)
        """
        if coef.dim() == 2 and coef.size(0) > 1:
            # Batched coefficients: shape (B, rank)
            # Compute a combined weight for each sample:
            # For each sample b: combined_weight[b] = sum_r coef[b, r] * self.weight[..., r]
            # Resulting shape: (B, out_features, in_features)
            combined_weight = torch.einsum("oir,br->boi", self.weight, coef)
            if self.bias is not None:
                combined_bias = torch.einsum("or,br->bo", self.bias, coef)
            else:
                combined_bias = None
            # Perform a per-sample linear transformation.
            # x: (B, in_features) -> unsqueeze to (B, 1, in_features)
            # combined_weight.transpose: (B, in_features, out_features)
            # bmm yields (B, 1, out_features) then squeeze to (B, out_features)
            output = torch.bmm(input.unsqueeze(1), combined_weight.transpose(1, 2)).squeeze(1)
            if combined_bias is not None:
                output = output + combined_bias
            return output
        else:
            # Global coefficients assumed to be shape (1, rank)
            coef = coef.squeeze(0)  # now shape: (rank,)
            combined_weight = torch.einsum("oir,r->oi", self.weight, coef)
            if self.bias is not None:
                combined_bias = torch.einsum("or,r->o", self.bias, coef)
            else:
                combined_bias = None
            return F.linear(input, combined_weight, combined_bias)
    

    def extra_repr(self) -> str:
        return (
            f"{self.in_features}, {self.out_features}, rank={self.rank}, "
            f"bias={self.bias is not None}"
        )


class MoA(nn.Module):
    """
    Mixture of MLP experts with input-dependent coefficients.
    
    Instead of a single global coefficient vector (as in MoE),
    this class uses a learnable matrix of shape (input_dim, rank). Given an input
    of shape (B, input_dim), multiplying by this matrix produces a coefficient tensor of shape (B, rank)
    so that every sample has its own set of coefficients.
    
    Args:
        input_dim (int): Dimension of the input data.
        hidden_dim (int): Dimension of the hidden layer.
        latent_dim (int): Dimension of the output latent embedding.
        dropout (float): Dropout probability.
        use_batch_norm (bool): Whether to apply batch normalization.
        use_layer_norm (bool): Whether to apply layer normalization.
        rank (int): Number of experts (rank dimension).
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 512,
                 latent_dim: int = 64,
                 dropout: float = 0.2,
                 use_batch_norm: bool = True,
                 use_layer_norm: bool = False,
                 decode_latent: bool = False,
                 rank: int = 16):
        super(MoA, self).__init__()
        self.rank = rank

        self.fc1 = MixtureLinear(input_dim, hidden_dim, rank=rank, bias=True)
        self.norm = (nn.BatchNorm1d(hidden_dim) if use_batch_norm
                     else nn.LayerNorm(hidden_dim) if use_layer_norm
                     else nn.Identity())
        self.act = nn.LeakyReLU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = MixtureLinear(hidden_dim, latent_dim, rank=rank, bias=True)
        self.drop2 = nn.Dropout(dropout)
        
        # Learnable coefficient matrix: maps input (of size input_dim) to coefficients (of size rank)
        # Shape: (input_dim, rank)
        self.coef_mat = nn.Parameter(torch.empty(input_dim, rank))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = (0.02 * self.rank**-0.5) ** 0.5
        trunc_normal_(self.coef_mat, std=std)
        # fc1 and fc2 initialize themselves

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, input_dim)
        Returns:
            Tensor of shape (B, latent_dim)
        """
        # Compute per-sample coefficients: (B, input_dim) x (input_dim, rank) -> (B, rank)
        coef = x @ self.coef_mat
        # (Optionally, you might normalize these coefficients, e.g., with softmax)
        coef = F.softmax(coef, dim=-1)
        
        # Pass the input along with its per-sample coefficients through the network.
        x = self.fc1(x, coef)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x, coef)
        x = self.drop2(x)
        
        return x, 0, 0













class MoE(nn.Module):
    """
    Mixture of MLP experts that mimics the interface of the existing MLP.
    It applies a mixture of linear layers with learnable coefficients (for a single token)
    to extract a latent representation.

    Args:
        input_dim (int): Dimension of the input data.
        hidden_dim (int): Dimension of the hidden layer.
        latent_dim (int): Dimension of the output latent embedding.
        dropout (float): Dropout probability.
        use_batch_norm (bool): Whether to apply batch normalization.
        use_layer_norm (bool): Whether to apply layer normalization.
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
        rank: int = 16,
    ):
        super(MoE, self).__init__()
        self.rank = rank

        self.fc1 = MixtureLinear(input_dim, hidden_dim, rank=rank, bias=True)
        self.norm = nn.BatchNorm1d(hidden_dim) if use_batch_norm else (
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
        )
        self.act = nn.LeakyReLU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = MixtureLinear(hidden_dim, latent_dim, rank=rank, bias=True)
        self.drop2 = nn.Dropout(dropout)
        # Learnable coefficient vector for the mixture; single token -> shape (rank,)
        self.coef = nn.Parameter(torch.empty(rank))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = (0.02 * self.rank**-0.5) ** 0.5
        trunc_normal_(self.coef.unsqueeze(0), std=std)
        # Reset parameters of fc1 and fc2 are handled inside their own reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: shape (batch, input_dim)
        coef_soft = F.softmax(self.coef, dim=0).unsqueeze(0) #self.coef.unsqueeze(0) #F.softmax(self.coef, dim=0).unsqueeze(0)
        x = self.fc1(x, coef_soft)  # Removed unsqueeze
        x = self.norm(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x, coef_soft)  # Removed unsqueeze here as well
        x = self.drop2(x)
        return x, 0, 0

if __name__ == "__main__":
    # Define input dimensions
    input_dim = 128
    hidden_dim = 512
    latent_dim = 64
    batch_size = 32

    # Create a random input tensor
    x = torch.randn(batch_size, input_dim)

    # Initialize the MoE model
    model = MoE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)

    # Forward pass
    output = model(x)

    # Print the output shape
    print("Output shape:", output.shape)