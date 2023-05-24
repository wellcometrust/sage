import torch.nn as nn
import torch
import tarfile


class Dummy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 1)

    def forward(self, x):
        return 1


model = Dummy()

# Save the model
torch.save(model.state_dict(), "dummy.pt")

with tarfile.open("dummy.pt.tar.gz", "w:gz") as tar:
    tar.add("dummy.pt")
