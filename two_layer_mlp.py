# training a two layer MLP for a regression problem 

import torch 
from torch.nn import Linear, ReLU
from torch.nn import MSELoss
import numpy as np

N,D_in,H,D_out = 1000,64,100,1
x = np.random.randn(N,D_in)
y = np.random.randn(N,D_out)

x = torch.from_numpy(x).float()
y = torch.from_numpy(y).float()

print (x.shape, y.shape)

# every layer takes the input and the output dimensions as well 

model = torch.nn.Sequential(
	Linear(D_in,H),
	ReLU(),
	Linear(H,D_out)
	)

loss_fn = MSELoss(reduction='sum')

learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for t in range(10000):

	y_pred = model(x)
	loss = loss_fn(y_pred, y)
	print("at iter ",t, " loss is ",loss.item())
	optimizer.zero_grad()
	loss.backward()
	optimizer.step() 

