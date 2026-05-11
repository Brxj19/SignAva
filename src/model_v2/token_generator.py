import torch
import torch.nn as nn


class MotionTokenGenerator(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        token_len: int = 20,
        codebook_size: int = 512,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_len = token_len
        self.codebook_size = codebook_size
        self.start_token_id = codebook_size
        self.gloss_embedding = nn.Embedding(vocab_size, d_model)
        self.token_embedding = nn.Embedding(codebook_size + 1, d_model)
        self.position_embedding = nn.Embedding(token_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, codebook_size)

    def forward(self, gloss_id: torch.Tensor, previous_tokens: torch.Tensor) -> torch.Tensor:
        if previous_tokens.ndim != 2:
            raise ValueError(f"Expected previous_tokens shape (batch, token_len), got {tuple(previous_tokens.shape)}")
        batch, token_len = previous_tokens.shape
        if token_len > self.token_len:
            raise ValueError(f"token_len {token_len} exceeds configured token_len {self.token_len}")

        positions = torch.arange(token_len, device=previous_tokens.device).unsqueeze(0).expand(batch, token_len)
        x = self.token_embedding(previous_tokens) + self.position_embedding(positions)
        x = x + self.gloss_embedding(gloss_id).unsqueeze(1)
        causal_mask = torch.triu(
            torch.ones(token_len, token_len, dtype=torch.bool, device=previous_tokens.device),
            diagonal=1,
        )
        x = self.transformer(x, mask=causal_mask)
        return self.output(self.norm(x))

    @torch.no_grad()
    def generate(self, gloss_id: torch.Tensor, temperature: float = 1.0, greedy: bool = True) -> torch.Tensor:
        batch = gloss_id.shape[0]
        tokens = torch.full((batch, 1), self.start_token_id, dtype=torch.long, device=gloss_id.device)
        generated: list[torch.Tensor] = []
        for _ in range(self.token_len):
            logits = self.forward(gloss_id, tokens)[:, -1, :]
            if greedy:
                next_token = torch.argmax(logits, dim=-1)
            else:
                probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            generated.append(next_token)
            tokens = torch.cat([tokens, next_token.unsqueeze(1)], dim=1)
        return torch.stack(generated, dim=1)
