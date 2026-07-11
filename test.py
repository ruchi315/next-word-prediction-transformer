import torch

print("Torch Version:", torch.__version__)
print("MPS Available:", torch.backends.mps.is_available())
print("CUDA Available:", torch.cuda.is_available())