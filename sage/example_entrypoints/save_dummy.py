import torch

from pt_dummy import Dummy

dummy_model = Dummy()
torch.save(dummy_model.state_dict(), "dummy.pt")
