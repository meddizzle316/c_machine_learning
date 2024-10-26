import gym
import numpy as np
from PIL import Image
import torch
import os

print(torch.cuda.is_available())

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# environment = DQNBreakout(device=device)