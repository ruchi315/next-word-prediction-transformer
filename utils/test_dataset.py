from utils.padding import padded_inputs
from utils.create_dataset import targets
from utils.dataset import NextWordDataset

dataset = NextWordDataset(
    padded_inputs,
    targets
)

print(dataset[0])