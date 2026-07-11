import torch
from models.transformer import GPT


MODEL_PATH = "checkpoints/model.pth"


device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)


model = None


def load_model():

    global model

    if model is None:

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


        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        model.eval()

        print("Model loaded successfully")


    return model, device