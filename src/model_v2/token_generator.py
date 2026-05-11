import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.gloss_projection = nn.Linear(d_model, d_model)
        nn.init.eye_(self.gloss_projection.weight)
        nn.init.zeros_(self.gloss_projection.bias)
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
        self.gloss_classifier = nn.Linear(d_model, vocab_size)

    def forward(self, gloss_id: torch.Tensor, previous_tokens: torch.Tensor) -> torch.Tensor:
        # Teacher forcing contract:
        # target tokens: [t0, t1, ..., t19]
        # previous_tokens input: [BOS, t0, ..., t18]
        # logits output predicts [t0, t1, ..., t19].
        if previous_tokens.ndim != 2:
            raise ValueError(f"Expected previous_tokens shape (batch, token_len), got {tuple(previous_tokens.shape)}")
        batch, token_len = previous_tokens.shape
        if token_len > self.token_len:
            raise ValueError(f"token_len {token_len} exceeds configured token_len {self.token_len}")
        if gloss_id.shape != (batch,):
            raise ValueError(f"Expected gloss_id shape ({batch},), got {tuple(gloss_id.shape)}")
        if previous_tokens.max().item() > self.start_token_id or previous_tokens.min().item() < 0:
            raise ValueError("previous_tokens contains ids outside [0, codebook_size] where codebook_size is BOS")

        hidden = self._hidden_states(gloss_id, previous_tokens)
        return self.output(hidden)

    def _hidden_states(self, gloss_id: torch.Tensor, previous_tokens: torch.Tensor) -> torch.Tensor:
        batch, token_len = previous_tokens.shape
        positions = torch.arange(token_len, device=previous_tokens.device).unsqueeze(0).expand(batch, token_len)
        gloss_condition = self.gloss_projection(self.gloss_embedding(gloss_id)).unsqueeze(1)
        x = self.token_embedding(previous_tokens) + self.position_embedding(positions) + gloss_condition
        causal_mask = torch.triu(
            torch.ones(token_len, token_len, dtype=torch.bool, device=previous_tokens.device),
            diagonal=1,
        )
        hidden = self.norm(self.transformer(x, mask=causal_mask))
        return hidden

    def forward_with_aux(self, gloss_id: torch.Tensor, previous_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self._hidden_states(gloss_id, previous_tokens)
        logits = self.output(hidden)
        gloss_logits = self.gloss_classifier(hidden.mean(dim=1))
        return logits, gloss_logits

    @torch.no_grad()
    def generate(
        self,
        gloss_id: torch.Tensor,
        temperature: float = 1.0,
        greedy: bool = True,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        top_k: int = 0,
    ) -> torch.Tensor:
        batch = gloss_id.shape[0]
        tokens = torch.full((batch, 1), self.start_token_id, dtype=torch.long, device=gloss_id.device)
        generated: list[torch.Tensor] = []
        for _ in range(self.token_len):
            logits = self.forward(gloss_id, tokens)[:, -1, :]
            logits = self._apply_repetition_controls(
                logits=logits,
                generated=generated,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
            if greedy:
                next_token = torch.argmax(logits, dim=-1)
            else:
                logits = logits / max(temperature, 1e-6)
                if top_k > 0 and top_k < logits.shape[-1]:
                    values, _ = torch.topk(logits, k=top_k, dim=-1)
                    cutoff = values[:, -1].unsqueeze(-1)
                    logits = torch.where(logits < cutoff, torch.full_like(logits, float("-inf")), logits)
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            generated.append(next_token)
            tokens = torch.cat([tokens, next_token.unsqueeze(1)], dim=1)
        return torch.stack(generated, dim=1)

    def _apply_repetition_controls(
        self,
        logits: torch.Tensor,
        generated: list[torch.Tensor],
        repetition_penalty: float,
        no_repeat_ngram_size: int,
    ) -> torch.Tensor:
        if not generated:
            return logits
        adjusted = logits.clone()
        history = torch.stack(generated, dim=1)
        if repetition_penalty > 1.0:
            for batch_index in range(history.shape[0]):
                used_tokens = torch.unique(history[batch_index])
                adjusted[batch_index, used_tokens] = adjusted[batch_index, used_tokens] / repetition_penalty
        if no_repeat_ngram_size > 0 and history.shape[1] >= no_repeat_ngram_size - 1:
            prefix_len = max(no_repeat_ngram_size - 1, 0)
            for batch_index in range(history.shape[0]):
                sequence = history[batch_index].tolist()
                if no_repeat_ngram_size == 1:
                    banned = set(sequence)
                else:
                    current_prefix = tuple(sequence[-prefix_len:])
                    banned = {
                        sequence[start + prefix_len]
                        for start in range(0, len(sequence) - no_repeat_ngram_size + 1)
                        if tuple(sequence[start : start + prefix_len]) == current_prefix
                    }
                if banned:
                    adjusted[batch_index, list(banned)] = float("-inf")
        return adjusted
