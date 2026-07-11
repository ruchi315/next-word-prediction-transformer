import os
import torch

from models.transformer import GPT
from utils.tokenizer import word2idx, idx2word


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
# Load Model
# =====================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "model.pth"
)

print("Loading model from:", MODEL_PATH)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model = GPT(
    vocab_size=checkpoint["vocab_size"],
    embed_dim=checkpoint["embed_dim"],
    num_heads=checkpoint["num_heads"],
    num_layers=checkpoint["num_layers"],
    max_length=checkpoint["max_length"]
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Model Loaded Successfully!\n")


# =====================================
# Generate Text
# =====================================

def generate_text(
    prompt,
    max_new_tokens=20,
    temperature=1.0,
    top_k=20
):

    words = prompt.lower().strip().split()

    tokens = []

    unknown = []

    for word in words:

        if word in word2idx:
            tokens.append(word2idx[word])
        else:
            unknown.append(word)

    if len(tokens) == 0:
        print("No known words found.")
        return

    if unknown:
        print("Unknown words ignored:", unknown)

    with torch.no_grad():

        for _ in range(max_new_tokens):

            input_tokens = tokens[-64:]

            x = torch.tensor(
                [input_tokens],
                dtype=torch.long,
                device=device
            )

            logits = model(x)

            logits = logits[:, -1, :] / temperature

            values, indices = torch.topk(
                logits,
                k=top_k
            )

            probs = torch.softmax(values, dim=-1)

            sampled = torch.multinomial(
                probs,
                num_samples=1
            )

            next_token = indices[0][sampled].item()

            tokens.append(next_token)

    sentence = []

    for token in tokens:
        sentence.append(
            idx2word.get(token, "<UNK>")
        )

    print("\nGenerated Text\n")
    print("-" * 60)
    print(" ".join(sentence))
    print("-" * 60)


# =====================================
# Interactive Loop
# =====================================

while True:

    prompt = input("\nEnter Prompt ('exit' to quit): ")

    if prompt.lower() == "exit":
        break

    generate_text(
        prompt,
        max_new_tokens=20,
        temperature=0.9,
        top_k=20
    )