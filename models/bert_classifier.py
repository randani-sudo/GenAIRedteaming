"""BERT-based binary classifier for response text."""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import config

class ResponseDataset(Dataset):
    def __init__(self, texts: list, labels: np.ndarray, tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tokenizer(
            self.texts[i],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[i], dtype=torch.long),
        }

class BERTClassifier(nn.Module):
    def __init__(self, bert_name: str = None, num_labels: int = 2, dropout: float = 0.1):
        super().__init__()
        bert_name = bert_name or config.BERT_MODEL
        self.bert = AutoModel.from_pretrained(bert_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0, :]
        x = self.dropout(pooled)
        return self.classifier(x)

def train_bert(
    train_texts: list,
    train_labels: np.ndarray,
    val_texts: list,
    val_labels: np.ndarray,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 2e-5,
    max_length: int = None,
):
    max_length = max_length or config.MAX_SEQ_LENGTH
    tokenizer = AutoTokenizer.from_pretrained(config.BERT_MODEL)
    train_ds = ResponseDataset(train_texts, train_labels, tokenizer, max_length)
    val_ds = ResponseDataset(val_texts, val_labels, tokenizer, max_length)
    device = torch.device("cuda" if (getattr(config, "USE_GPU", True) and torch.cuda.is_available()) else "cpu")
    # num_workers=0 to avoid DataLoader hang on some Linux/DGX setups; pin_memory helps GPU transfer
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    model = BERTClassifier().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            opt.zero_grad()
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            loss = criterion(logits, batch["labels"].to(device))
            loss.backward()
            opt.step()
        model.eval()
        val_loss = 0.0
        preds, truths = [], []
        with torch.no_grad():
            for batch in val_loader:
                logits = model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                )
                val_loss += criterion(logits, batch["labels"].to(device)).item()
                preds.extend(logits.argmax(1).cpu().numpy())
                truths.extend(batch["labels"].numpy())
        from sklearn.metrics import f1_score
        f1 = f1_score(truths, preds, zero_division=0)
        print(f"BERT Epoch {epoch+1}/{epochs} val_loss={val_loss/len(val_loader):.4f} val_f1={f1:.4f}")
    print("BERT training done.")
    return model, tokenizer  # (BERTClassifier, PreTrainedTokenizer)

def predict_bert(model, tokenizer, texts: list, batch_size: int = 32, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=config.MAX_SEQ_LENGTH,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        all_probs.append(probs)
    return np.concatenate(all_probs) if all_probs else np.array([])

def save_bert(model, tokenizer, path: Path) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / "bert_classifier.pt")
    tokenizer.save_pretrained(path)

def load_bert(path: Path, device=None) -> tuple:
    path = Path(path)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(path))
    model = BERTClassifier()
    model.load_state_dict(torch.load(path / "bert_classifier.pt", map_location=device))
    model.to(device)
    return model, tokenizer
