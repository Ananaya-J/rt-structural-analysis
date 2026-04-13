"""
model.py — DomainPredictor model definitions
Importable by both 03_train_model.py and 04_evaluate.py
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class DomainDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        return {
            "seq": torch.tensor(d["seq_encoded"], dtype=torch.long),
            "labels": torch.tensor(d["labels_encoded"], dtype=torch.long),
            "length": d["length"],
            "accession": d["accession"],
        }


def collate_fn(batch, pad_idx=21, pad_label_idx=-100):
    seqs = [b["seq"] for b in batch]
    labels = [b["labels"] for b in batch]
    lengths = [b["length"] for b in batch]
    accessions = [b["accession"] for b in batch]

    sorted_idx = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    seqs = [seqs[i] for i in sorted_idx]
    labels = [labels[i] for i in sorted_idx]
    lengths = [lengths[i] for i in sorted_idx]
    accessions = [accessions[i] for i in sorted_idx]

    seqs_padded = pad_sequence(seqs, batch_first=True, padding_value=pad_idx)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=pad_label_idx)

    return {
        "seq": seqs_padded,
        "labels": labels_padded,
        "lengths": lengths,
        "accessions": accessions,
    }


class CRF(nn.Module):
    def __init__(self, num_tags):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))

    def forward(self, emissions, tags, mask):
        gold_score = self._score_sentence(emissions, tags, mask)
        forward_score = self._forward_algorithm(emissions, mask)
        return (forward_score - gold_score).mean()

    def _forward_algorithm(self, emissions, mask):
        batch_size, seq_len, num_tags = emissions.shape
        score = self.start_transitions + emissions[:, 0]
        for i in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)
            broadcast_emission = emissions[:, i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emission
            next_score = torch.logsumexp(next_score, dim=1)
            mask_i = mask[:, i].unsqueeze(1)
            score = torch.where(mask_i.bool(), next_score, score)
        score = score + self.end_transitions
        return torch.logsumexp(score, dim=1)

    def _score_sentence(self, emissions, tags, mask):
        batch_size, seq_len, num_tags = emissions.shape
        score = self.start_transitions[tags[:, 0]]
        score += emissions[:, 0].gather(1, tags[:, 0].unsqueeze(1)).squeeze(1)
        for i in range(1, seq_len):
            mask_i = mask[:, i]
            trans_score = self.transitions[tags[:, i], tags[:, i - 1]]
            emit_score = emissions[:, i].gather(1, tags[:, i].unsqueeze(1)).squeeze(1)
            score += (trans_score + emit_score) * mask_i
        seq_lengths = mask.long().sum(dim=1) - 1
        last_tags = tags.gather(1, seq_lengths.unsqueeze(1)).squeeze(1)
        score += self.end_transitions[last_tags]
        return score

    def decode(self, emissions, mask):
        batch_size, seq_len, num_tags = emissions.shape
        score = self.start_transitions + emissions[:, 0]
        history = []
        for i in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)
            broadcast_emission = emissions[:, i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emission
            next_score, indices = next_score.max(dim=1)
            mask_i = mask[:, i].unsqueeze(1).bool()
            score = torch.where(mask_i, next_score, score)
            history.append(indices)
        score += self.end_transitions
        seq_lengths = mask.long().sum(dim=1) - 1
        best_tags_list = []
        for b in range(batch_size):
            best_last_tag = score[b].argmax().item()
            best_tags = [best_last_tag]
            for hist in reversed(history[:seq_lengths[b].item()]):
                best_last_tag = hist[b][best_last_tag].item()
                best_tags.append(best_last_tag)
            best_tags.reverse()
            best_tags_list.append(best_tags)
        return best_tags_list


class DomainPredictor(nn.Module):
    def __init__(self, vocab_size=22, embed_dim=64, hidden_dim=256,
                 num_layers=2, num_classes=6, dropout=0.3,
                 use_crf=True, pad_idx=21):
        super().__init__()
        self.use_crf = use_crf
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)
        if use_crf:
            self.crf = CRF(num_classes)
        else:
            self.crf = None

    def _get_emissions(self, seq, lengths):
        embedded = self.dropout(self.embedding(seq))
        packed = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=True)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        return self.classifier(self.dropout(lstm_out))

    def forward(self, seq, labels, lengths):
        emissions = self._get_emissions(seq, lengths)
        batch_size, max_len = seq.shape
        mask = torch.zeros(batch_size, max_len, device=seq.device)
        for i, l in enumerate(lengths):
            mask[i, :l] = 1.0
        if self.use_crf:
            safe_labels = labels.clone()
            safe_labels[safe_labels == -100] = 0
            return self.crf(emissions, safe_labels, mask)
        else:
            return nn.functional.cross_entropy(
                emissions.view(-1, emissions.size(-1)),
                labels.view(-1), ignore_index=-100,
            )

    def predict(self, seq, lengths):
        emissions = self._get_emissions(seq, lengths)
        batch_size, max_len = seq.shape
        mask = torch.zeros(batch_size, max_len, device=seq.device)
        for i, l in enumerate(lengths):
            mask[i, :l] = 1.0
        if self.use_crf:
            return self.crf.decode(emissions, mask)
        else:
            return emissions.argmax(dim=-1).cpu().tolist()
