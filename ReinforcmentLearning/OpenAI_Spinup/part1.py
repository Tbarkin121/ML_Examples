import torch
from torch import nn

obs_dim=4
act_dim = 4
pi_net = nn.Sequential( 
              nn.Linear(obs_dim, 64),
              nn.Tanh(),
              nn.Linear(64, 64),
              nn.Tanh(),
              nn.Linear(64, act_dim)
            )

# model = NeuralNetwork().to(device)
print(pi_net)

#%%
obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
actions = pi_net(obs_tensor)


