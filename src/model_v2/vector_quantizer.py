import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        codebook_size: int = 512,
        embedding_dim: int = 256,
        commitment_beta: float = 0.25,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.commitment_beta = commitment_beta
        self.embedding = nn.Embedding(codebook_size, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if z.ndim != 3:
            raise ValueError(f"Expected z shape (batch, tokens, dim), got {tuple(z.shape)}")
        if z.shape[-1] != self.embedding_dim:
            raise ValueError(f"Expected embedding dim {self.embedding_dim}, got {z.shape[-1]}")

        flat_z = z.reshape(-1, self.embedding_dim)
        codebook = self.embedding.weight
        distances = (
            flat_z.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * flat_z @ codebook.t()
            + codebook.pow(2).sum(dim=1).unsqueeze(0)
        )
        flat_indices = torch.argmin(distances, dim=1)
        code_indices = flat_indices.view(z.shape[0], z.shape[1])
        quantized = self.embedding(flat_indices).view_as(z)

        codebook_loss = F.mse_loss(quantized, z.detach())
        commitment_loss = F.mse_loss(z, quantized.detach())
        vq_loss = codebook_loss + self.commitment_beta * commitment_loss
        quantized = z + (quantized - z).detach()

        one_hot = F.one_hot(flat_indices, self.codebook_size).to(z.dtype)
        avg_probs = one_hot.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        usage = (avg_probs > 0).sum()
        stats = {
            "codebook_usage": usage,
            "codebook_usage_fraction": usage.to(z.dtype) / float(self.codebook_size),
        }
        return quantized, code_indices, vq_loss, perplexity, stats

    def codes_to_embeddings(self, code_indices: torch.Tensor) -> torch.Tensor:
        return self.embedding(code_indices)
