import torch

from app.model_loader import load_model
from utils.tokenizer import word2idx, idx2word


def predict_next_words(
        text,
        max_new_tokens=3,
        temperature=0.8,
        top_k=20
):

    model, device = load_model()

    words = text.lower().strip().split()

    tokens = []

    # Convert words to tokens
    # Unknown words are converted to <UNK>
    for word in words:

        tokens.append(
            word2idx.get(
                word,
                word2idx["<UNK>"]
            )
        )


    if len(tokens) == 0:

        return [
            "No input detected"
        ]


    generated_tokens = tokens.copy()


    with torch.no_grad():

        for _ in range(max_new_tokens):

            input_tokens = generated_tokens[-64:]


            x = torch.tensor(
                [input_tokens],
                dtype=torch.long,
                device=device
            )


            logits = model(x)


            # Take last word prediction
            logits = logits[:, -1, :]


            # Temperature scaling
            logits = logits / temperature


            # Get top probable words
            values, indices = torch.topk(
                logits,
                k=top_k
            )


            # Pick highest probability word
            probs = torch.softmax(
                values,
                dim=-1
            )

            sample = torch.multinomial(
                probs,
                num_samples=1
            )

            next_token = indices[0][sample].item()


            generated_tokens.append(
                next_token
            )


    # Convert tokens back to words

    generated_words = []

    for token in generated_tokens:

        generated_words.append(
            idx2word.get(
                token,
                "<UNK>"
            )
        )


    # Return only newly generated words
    new_words = generated_words[len(words):]


    # Remove duplicates
    unique_words = []

    for word in new_words:

        if word not in unique_words:

            unique_words.append(word)


    return unique_words