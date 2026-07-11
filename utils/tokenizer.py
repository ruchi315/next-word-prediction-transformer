from datasets import load_dataset

dataset = load_dataset(
    "wikitext",
    "wikitext-2-raw-v1"
)

train_sentences = []

for sample in dataset["train"]:

    text = sample["text"].strip()

    if len(text) > 0 and not text.startswith("="):
        train_sentences.append(text)
vocab = set()

for sentence in train_sentences:

    words = sentence.lower().split()

    vocab.update(words)

print("Vocabulary Size:", len(vocab))
# Create dictionaries

word2idx = {
    "<PAD>":0,
    "<UNK>":1
}

idx2word = {
    0:"<PAD>",
    1:"<UNK>"
}

for index, word in enumerate(sorted(vocab), start=2):

    word2idx[word] = index
    idx2word[index] = word
# Encode every sentence into integers

encoded_sentences = []

for sentence in train_sentences:

    words = sentence.lower().split()

    encoded = []

    for word in words:
        encoded.append(
            word2idx.get(
                word,
                word2idx["<UNK>"]
            )
        )

    encoded_sentences.append(encoded)
print("Vocabulary Size:", len(word2idx))

print("\nOriginal Sentence:")
print(train_sentences[0])

print("\nEncoded Sentence:")
print(encoded_sentences[0])
idx2word = {v: k for k, v in word2idx.items()}