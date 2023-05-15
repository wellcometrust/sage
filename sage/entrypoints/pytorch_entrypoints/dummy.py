import torch.nn as nn
import os
import torch
import json


class Dummy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 1)

    def forward(self, x):
        return 1


def model_fn(model_path: str):
    model = Dummy()
    state_dict_path = os.path.join(model_path, "dummy.pt")
    model.load_state_dict(torch.load(state_dict_path))
    return model


def input_fn(input_data, content_type="application/json"):
    _ = json.loads(input_data)
    # Endpoint expects tensor as input. Send dummy tensor
    return torch.tensor([1, 2, 3])


def predict_pytorch(text, model_path):
    model = model_fn(model_path)
    prediction = model.predict([text])
    return prediction[0]
