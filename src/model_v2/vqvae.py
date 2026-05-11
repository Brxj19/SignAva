import torch
import torch.nn as nn

from src.model_v2.vector_quantizer import VectorQuantizer


class MotionVQVAE(nn.Module):
    def __init__(
        self,
        seq_len: int = 80,
        motion_dim: int = 182,
        latent_dim: int = 256,
        codebook_size: int = 512,
        downsample_factor: int = 4,
        commitment_beta: float = 0.25,
        dropout: float = 0.1,
    ):
        super().__init__()
        if seq_len % downsample_factor != 0:
            raise ValueError("seq_len must be divisible by downsample_factor")
        if downsample_factor != 4:
            raise ValueError("This implementation currently supports downsample_factor=4")

        self.seq_len = seq_len
        self.motion_dim = motion_dim
        self.latent_dim = latent_dim
        self.codebook_size = codebook_size
        self.downsample_factor = downsample_factor
        self.token_len = seq_len // downsample_factor

        self.encoder = nn.Sequential(
            nn.Conv1d(motion_dim, latent_dim, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(latent_dim, latent_dim, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(latent_dim, latent_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.quantizer = VectorQuantizer(
            codebook_size=codebook_size,
            embedding_dim=latent_dim,
            commitment_beta=commitment_beta,
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(latent_dim, latent_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(latent_dim, latent_dim, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.ConvTranspose1d(latent_dim, latent_dim, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(latent_dim, motion_dim, kernel_size=5, padding=2),
        )

    def encode(self, motion: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if motion.ndim != 3:
            raise ValueError(f"Expected motion shape (batch, seq_len, motion_dim), got {tuple(motion.shape)}")
        features = self.encoder(motion.transpose(1, 2)).transpose(1, 2)
        quantized, code_indices, _, _, _ = self.quantizer(features)
        return code_indices, quantized, features

    def decode(self, code_indices: torch.Tensor | None = None, quantized: torch.Tensor | None = None) -> torch.Tensor:
        if quantized is None:
            if code_indices is None:
                raise ValueError("decode requires code_indices or quantized")
            quantized = self.quantizer.codes_to_embeddings(code_indices)
        recon = self.decoder(quantized.transpose(1, 2)).transpose(1, 2)
        if recon.shape[1] != self.seq_len:
            recon = recon[:, : self.seq_len, :]
        return recon

    def forward(self, motion: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.encoder(motion.transpose(1, 2)).transpose(1, 2)
        quantized, code_indices, vq_loss, perplexity, _ = self.quantizer(features)
        recon = self.decode(quantized=quantized)
        return recon, code_indices, vq_loss, perplexity
