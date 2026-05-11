import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_model_from_config(config: dict[str, Any]) -> "SignMotionGenerator":
    return SignMotionGenerator(
        vocab_size=config["vocab_size"],
        seq_len=config.get("seq_len", 60),
        motion_dim=config.get("motion_dim", 182),
        d_model=config.get("d_model", 256),
        nhead=config.get("nhead", 8),
        num_layers=config.get("num_layers", 4),
        dropout=config.get("dropout", 0.1),
    )


class SignMotionGenerator(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int = 60,
        motion_dim: int = 182,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.motion_dim = motion_dim
        self.d_model = d_model

        # Gloss embedding
        self.gloss_embedding = nn.Embedding(vocab_size, d_model)

        # Temporal queries: learned embeddings for each frame position
        self.temporal_queries = nn.Embedding(seq_len, d_model)

        # Positional encoding: sinusoidal
        self.register_buffer("pos_encoding", self._create_sinusoidal_pos_encoding(seq_len, d_model))

        # Transformer Decoder
        decoder_layer = TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output projection
        self.output_projection = nn.Linear(d_model, motion_dim)

    def _create_sinusoidal_pos_encoding(self, seq_len: int, d_model: int) -> torch.Tensor:
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # Shape: (1, seq_len, d_model)

    def forward(self, gloss_id: torch.Tensor) -> torch.Tensor:
        batch_size = gloss_id.size(0)
        device = gloss_id.device
        if gloss_id.dim() != 1:
            raise ValueError(f"gloss_id must be 1D tensor, got shape {gloss_id.shape}")

        # Gloss embedding: (batch,) -> (batch, d_model)
        gloss_emb = self.gloss_embedding(gloss_id)  # (batch, d_model)

        # Temporal queries: (seq_len, d_model) -> (batch, seq_len, d_model)
        temporal_queries = self.temporal_queries.weight.unsqueeze(0).expand(batch_size, -1, -1)

        # Add positional encoding (already on device via register_buffer)
        temporal_queries = temporal_queries + self.pos_encoding.expand(batch_size, -1, -1)

        # Memory for decoder: gloss_emb repeated for each position
        memory = gloss_emb.unsqueeze(1).expand(-1, self.seq_len, -1)  # (batch, seq_len, d_model)

        # Transformer decoder: tgt is temporal_queries, memory is gloss_emb
        # Note: TransformerDecoder expects tgt and memory of shape (batch, seq_len, d_model)
        output = self.transformer_decoder(temporal_queries, memory)  # (batch, seq_len, d_model)

        # Output projection
        motion = self.output_projection(output)  # (batch, seq_len, motion_dim)

        if motion.shape != (batch_size, self.seq_len, self.motion_dim):
            raise RuntimeError(f"Unexpected output shape: {motion.shape}, expected {(batch_size, self.seq_len, self.motion_dim)}")

        return motion


if __name__ == "__main__":
    # Self-test
    model = SignMotionGenerator(vocab_size=12)
    print(f"Model parameters: {count_parameters(model)}")

    # Test on CPU
    print("\nTesting on CPU:")
    gloss_id_cpu = torch.randint(0, 12, (4,))  # (batch,)
    print(f"Input gloss_id shape: {gloss_id_cpu.shape}")
    output_cpu = model(gloss_id_cpu)
    print(f"Output shape: {output_cpu.shape}")
    print(f"Output device: {output_cpu.device}")
    assert output_cpu.shape == (4, 60, 182), f"Expected (4, 60, 182), got {output_cpu.shape}"
    assert output_cpu.device == torch.device("cpu"), f"Expected CPU, got {output_cpu.device}"

    # Test on CUDA if available
    if torch.cuda.is_available():
        print("\nTesting on CUDA:")
        model_cuda = model.to("cuda")
        gloss_id_cuda = gloss_id_cpu.to("cuda")
        output_cuda = model_cuda(gloss_id_cuda)
        print(f"Output shape: {output_cuda.shape}")
        print(f"Output device: {output_cuda.device}")
        assert output_cuda.shape == (4, 60, 182), f"Expected (4, 60, 182), got {output_cuda.shape}"
        assert output_cuda.device == torch.device("cuda"), f"Expected CUDA, got {output_cuda.device}"
        print("CUDA test passed.")
    else:
        print("\nCUDA not available, skipping CUDA test.")

    # Loss test
    target = torch.randn(4, 60, 182)
    from src.model.loss import SignMotionLoss
    loss_fn = SignMotionLoss()
    loss_dict = loss_fn(output_cpu, target)
    print("\nLoss components:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value.item():.4f}")
