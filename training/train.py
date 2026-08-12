import torch
from torch.utils.data import DataLoader, random_split
from utils.padding import padded_inputs, targets
from utils.dataset import NextWordDataset
from utils.tokenizer import word2idx
from models.transformer import GPT
import os

MAX_EXAMPLES = 100000
full_dataset = NextWordDataset(
    padded_inputs[:MAX_EXAMPLES],
    targets[:MAX_EXAMPLES]
)

# ===== Train/Val split (90/10, matches your README claim) =====
val_size = int(0.1 * len(full_dataset))
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, drop_last=False)

print(f"Train samples: {train_size} | Val samples: {val_size}")

# ... (device, model, criterion, optimizer setup stays the same) ...

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

    # ===== Validation loop (no gradients, no weight updates) =====
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)[:, -1, :]
            loss = criterion(output, y)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)

    print(f"\nEpoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Save best checkpoint based on val loss, not just every epoch
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "vocab_size": vocab_size,
            "embed_dim": 256, "num_heads": 4, "num_layers": 4, "max_length": 64,
            "val_loss": avg_val_loss
        }
        torch.save(checkpoint, "best_model.pth")
        print(f"New best model saved (val loss: {avg_val_loss:.4f})")

print(f"\nTraining completed! Best Validation Loss: {best_val_loss:.4f}")
