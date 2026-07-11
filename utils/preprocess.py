from datasets import load_dataset

dataset = load_dataset(
    "wikitext",
    "wikitext-2-raw-v1"
)

train_sentences = []

for sample in dataset["train"]:

    text = sample["text"].strip()

    if len(text) > 0:

        train_sentences.append(text)

print("Total sentences:", len(train_sentences))

print(train_sentences[:10])