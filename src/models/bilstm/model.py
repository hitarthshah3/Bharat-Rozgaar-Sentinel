import torch
from torch import nn


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
                 pad_id: int = 0, dropout: float = 0.35, pretrained=None,
                 metadata_dim: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        if pretrained is not None:
            self.embedding.weight.data.copy_(torch.as_tensor(pretrained, dtype=torch.float32))
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True,
                            bidirectional=True, dropout=0.0)
        self.dropout = nn.Dropout(dropout)
        self.metadata_dim = metadata_dim
        if metadata_dim:
            self.metadata_encoder = nn.Sequential(
                nn.Linear(metadata_dim, 64), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(64, 32), nn.ReLU(),
            )
        self.classifier = nn.Linear(hidden_dim * 2 + (32 if metadata_dim else 0), 1)

    def forward(self, token_ids, lengths=None, metadata=None):
        embedded = self.embedding(token_ids)
        encoded, _ = self.lstm(embedded)
        if lengths is not None:
            mask = torch.arange(encoded.size(1), device=encoded.device)[None, :] < lengths[:, None]
            encoded = encoded * mask.unsqueeze(-1)
            pooled = encoded.sum(dim=1) / lengths.clamp(min=1).unsqueeze(1)
        else:
            pooled = encoded.mean(dim=1)
        features = self.dropout(pooled)
        if self.metadata_dim:
            if metadata is None:
                raise ValueError("Metadata input is required for this hybrid model.")
            features = torch.cat((features, self.metadata_encoder(metadata)), dim=1)
        return self.classifier(features).squeeze(-1)

