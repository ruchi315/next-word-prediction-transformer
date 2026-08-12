import torch
from torch.utils.data import DataLoader, random_split
from utils.padding import padded_inputs, targets
from utils.dataset import NextWordDataset
from utils.tokenizer import word2idx
from models.transformer import GPT
import os
import math

print("Current Working Directory:", os.getcwd())

MAX_EXAMPLES = 100000
full_dataset = NextWordDataset(
    padded_inputs[:MAX_EXAMPLES],
    targets[:MAX_EXAMPLES]
)

val_size = int(0.1 * len(full_dataset))
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, drop_last=False)

print(f"Train samples: {train_size} | Val samples: {val_size}")

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print("Using device:", device)

vocab_size = len(word2idx)
print("Vocabulary:", vocab_size)

model = GPT(
    vocab_size=vocab_size,
    embed_dim=256,
    num_heads=4,
    num_layers=4,
    max_length=64
).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

best_val_loss = float("inf")
epochs = 5

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)[:, -1, :]
        loss = criterion(output, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

    avg_train_loss = total_loss / len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)[:, -1, :]
            loss = criterion(output, y)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)

    print(f"\nEpoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}\n")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "vocab_size": vocab_size,
        "embed_dim": 256, "num_heads": 4, "num_layers": 4, "max_length": 64,
        "val_loss": avg_val_loss
    }
    torch.save(checkpoint, f"model_epoch_{epoch+1}.pth")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(checkpoint, "best_model.pth")

perplexity = math.exp(best_val_loss)
print(f"Training completed! Best Validation Loss: {best_val_loss:.4f}")
print(f"Validation Perplexity: {perplexity:.2f}")
