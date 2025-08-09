import sys

def show(label, fn):
    try:
        v = fn()
        print(f"{label}: {v}")
    except Exception as e:
        print(f"{label}: ERROR -> {e}")

print("Python exe:", sys.executable)
print("Python ver:", sys.version.split()[0])

# Torch (optional if not installed yet)
def _torch_ver():
    import torch
    return f"{torch.__version__} | CUDA available? {torch.cuda.is_available()}"
show("torch", _torch_ver)

def _gym_ver():
    import gymnasium as gym
    return gym.__version__
show("gymnasium", _gym_ver)

def _pz_ver():
    import pettingzoo as pz
    return pz.__version__
show("pettingzoo", _pz_ver)

def _ss_ver():
    import supersuit
    return supersuit.__version__
show("supersuit", _ss_ver)

def _trl_td_ver():
    import torchrl, tensordict
    return f"torchrl {torchrl.__version__}, tensordict {tensordict.__version__}"
show("torchrl/tensordict", _trl_td_ver)

print("✓ env check finished.")
import sys, torch, gymnasium, pettingzoo, supersuit, torchrl, tensordict

print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("Torch:", torch.__version__, "CUDA available?", torch.cuda.is_available())
print("Gymnasium:", gymnasium.__version__)
print("PettingZoo:", pettingzoo.__version__)
print("SuperSuit:", supersuit.__version__)
print("TorchRL:", torchrl.__version__)
print("TensorDict:", tensordict.__version__)
