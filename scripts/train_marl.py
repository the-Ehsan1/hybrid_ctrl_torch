import sys
print("Python:", sys.version)
import torch, gymnasium, pettingzoo, supersuit, torchrl, tensordict
print("OK:",
      "torch", torch.__version__,
      "gymnasium", gymnasium.__version__,
      "pettingzoo", pettingzoo.__version__,
      "supersuit", supersuit.__version__,
      "torchrl", torchrl.__version__,
      "tensordict", tensordict.__version__)
