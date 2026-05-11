import torch
import torch.nn as nn

from src.model_v2.vector_quantizer import EMAVectorQuantizer, VectorQuantizer


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
        quantizer_type: str = "ema",
        disable_quantization: bool = False,
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
        self.quantizer_type = quantizer_type
        self.disable_quantization = disable_quantization
        if quantizer_type not in {"standard", "ema"}:
            raise ValueError("quantizer_type must be one of: standard, ema")

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
        quantizer_cls = EMAVectorQuantizer if quantizer_type == "ema" else VectorQuantizer
        self.quantizer = quantizer_cls(codebook_size=codebook_size, embedding_dim=latent_dim, commitment_beta=commitment_beta)
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
        if self.disable_quantization:
            code_indices = torch.zeros(features.shape[:2], dtype=torch.long, device=features.device)
            return code_indices, features, features
        quantized, code_indices, _, _, _ = self.quantizer(features)
        return code_indices, quantized, features

    def decode(self, code_indices: torch.Tensor | None = None, quantized: torch.Tensor | None = None) -> torch.Tensor:
        if quantized is None:
            if code_indices is None:
                raise ValueError("decode requires code_indices or quantized")
            if self.disable_quantization:
                raise ValueError("decode with code_indices is unavailable when quantization is disabled")
            quantized = self.quantizer.codes_to_embeddings(code_indices)
        recon = self.decoder(quantized.transpose(1, 2)).transpose(1, 2)
        if recon.shape[1] != self.seq_len:
            recon = recon[:, : self.seq_len, :]
        return recon

    def forward(self, motion: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.encoder(motion.transpose(1, 2)).transpose(1, 2)
        self._last_encoder_features = features.detach()
        if self.disable_quantization:
            code_indices = torch.zeros(features.shape[:2], dtype=torch.long, device=features.device)
            recon = self.decode(quantized=features)
            zero = motion.new_tensor(0.0)
            return recon, code_indices, zero, zero
        quantized, code_indices, vq_loss, perplexity, _ = self.quantizer(features)
        recon = self.decode(quantized=quantized)
        return recon, code_indices, vq_loss, perplexity

    @torch.no_grad()
    def reset_dead_codes(self, code_indices: torch.Tensor) -> int:
        if self.disable_quantization:
            return 0
        if not hasattr(self, "_last_encoder_features"):
            return 0
        return self.quantizer.reset_codes(code_indices, self._last_encoder_features)
