import torch.nn as nn
import torch
import tarfile
import os
import shutil
from sage import ROOT_DIR


class Dummy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 1)

    def forward(self, x):
        return 1


model = Dummy()

# Create a dir
if os.path.exists('dummy'):
    shutil.rmtree('dummy')

os.mkdir('dummy')
os.mkdir(os.path.join('dummy', 'code'))

# Save the model
torch.save(model.state_dict(), os.path.join("dummy", "dummy.pt"))

shutil.copy(os.path.join(ROOT_DIR, "handlers", "pytorch", "inference.py"), os.path.join("dummy", "code"))

with tarfile.open("dummy.pt.tar.gz", "w:gz") as tar:
    tar.add(os.path.join('dummy', 'dummy.pt'), arcname='dummy.pt')
    tar.add(os.path.join('dummy', 'code'), arcname='code')
