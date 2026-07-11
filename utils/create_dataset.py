from utils.tokenizer import encoded_sentences

CONTEXT_SIZE = 64

inputs = []
targets = []

for sentence in encoded_sentences:

    if len(sentence) < 2:
        continue

    for i in range(1, len(sentence)):

        start = max(0, i - CONTEXT_SIZE)

        context = sentence[start:i]

        inputs.append(context)

        targets.append(sentence[i])

print("Context Size:", CONTEXT_SIZE)
print("Total Training Examples:", len(inputs))

print("\nExample Input:")
print(inputs[0])

print("\nExample Target:")
print(targets[0])