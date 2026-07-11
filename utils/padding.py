import torch
from torch.nn.utils.rnn import pad_sequence

from utils.create_dataset import inputs, targets

# Pad all input sequences to the same length
padded_inputs = pad_sequence(
    [torch.tensor(x, dtype=torch.long) for x in inputs],
    batch_first=True,
    padding_value=0  # <PAD> token index
)

targets = torch.tensor(
    targets,
    dtype=torch.long
)

print("Input Shape :", padded_inputs.shape)
print("Target Shape:", targets.shape)