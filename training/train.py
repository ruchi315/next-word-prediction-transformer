import torch
from torch.utils.data import DataLoader

from utils.padding import padded_inputs, targets
from utils.dataset import NextWordDataset
from utils.tokenizer import word2idx

from models.transformer import GPT
import os

print("Current Working Directory:", os.getcwd())
print("Running File:", __file__)

# =====================================
# Dataset
# =====================================

MAX_EXAMPLES = 100000       # Increase later to 100000 or full dataset

dataset = NextWordDataset(
    padded_inputs[:MAX_EXAMPLES],
    targets[:MAX_EXAMPLES]
)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    drop_last=True
)


# =====================================
# Device
# =====================================

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)


# =====================================
# Model
# =====================================

vocab_size = len(word2idx)

print("Vocabulary:", vocab_size)

model = GPT(
    vocab_size=vocab_size,
    embed_dim=256,
    num_heads=4,
    num_layers=4,
    max_length=64
).to(device)


# =====================================
# Loss
# =====================================

criterion = torch.nn.CrossEntropyLoss()


# =====================================
# Optimizer
# =====================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=0.01
)


# =====================================
# Training
# =====================================

epochs = 5


for epoch in range(epochs):

    model.train()

    total_loss = 0.0

    for batch_idx, (x, y) in enumerate(loader):

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        output = model(x)

        output = output[:, -1, :]

        loss = criterion(output, y)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print(
                f"Epoch {epoch+1} | Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.4f}"
            )

    avg_loss = total_loss / len(loader)

    print(
        f"\nEpoch {epoch+1}/{epochs} Average Loss: {avg_loss:.4f}"
    )

    # =====================================
    # Save checkpoint after every epoch
    # =====================================

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "vocab_size": vocab_size,
        "embed_dim": 256,
        "num_heads": 4,
        "num_layers": 4,
        "max_length": 64
    }
    print("About to save model...")
    print("Saving to:", os.path.abspath("model.pth"))

    torch.save(
        checkpoint,
        f"model_epoch_{epoch+1}.pth"
    )

    torch.save(
        checkpoint,
        "model.pth"
    )

    print(f"Model saved after epoch {epoch+1}\n")


print("Training completed successfully!")