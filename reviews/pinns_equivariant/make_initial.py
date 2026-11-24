"""Materialize the one seed-42 initialization shared across all arms and hosts; no optimizer."""
from pathlib import Path
import numpy as np
import torch
from equivariant_model import GroundForecaster
p=Path(__file__).resolve().parent/'initial_state.npz'
torch.manual_seed(42)
state=GroundForecaster(False,False).state_dict()
np.savez_compressed(p,**{k:tensor.numpy() for k,tensor in state.items()})
