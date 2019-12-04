import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Note_CNN(nn.Module):
	def __init__(self, num_steps_input, n_instruments):
    	super(Note_CNN, self).__init__()

	    #L_out = [L_in - kernel_size + 1] - for  L_out = L_in/1, use kernel size of (num_steps_input/2)+1
	    self.convs = nn.Conv1d(in_channels=n_instruments, out_channels=64, kernel_size=int(num_steps_input/2)+1)

	    self.relu1 = nn.ReLU()
	    
	    #flatten all the output channels into one - new size is 64*L_in/2  
	    self.flatten = nn.Flatten()

	    #linearly combine input to produce 32 outputs
	    self.linear = nn.Linear(in_features=64*int(n_instruments/2), out_features=32)

	    self.relu2 = nn.ReLU()

	    #output layer, converts 32 values from previous layer into outputs for each category
	    self.output = (in_features=32, out_features=n_instruments)

	    self.softmax = nn.Softmax()

	def forward(self, x):
	    output = self.convs(x)
	    output = self.relu1(output)
	    output = self.flatten(output)
	    output = self.linear(output)
	    output = self.relu2(output)
	    output = self.output(output)
	    output = self.softmax(output)

	    return output


X = np.zeros((32, 10))
X2 = np.zeros((32, 10))

for i in range(32):
	X[i,:] = [1,0,0,0,0,0,0,0,0,0]
	X2[i,:] = [0,0,0,0,0,0,0,0,0,1]

X = Variable(torch.tensor(X))
X.requires_grad = False

X2 = Variable(torch.tensor(X2))
X2.requires_grad = False

y = Variable(torch.tensor([1,0,0,0,0,0,0,0,0,0]))
y.requires_grad = False

y2 = Variable(torch.tensor([0,0,0,0,0,0,0,0,0,1])) 
y2.requires_grad = False

net = Note_CNN(32, 10)

optim = torch.optim.Adam(net.parameters(), lr=1e-3)

crossentropy = torch.nn.CrossEntropyLoss()

for i in range(100):
	optim.zero_grad()

	if (i%2==0):
		loss = crossentropy(net(X), y)
	else:
		loss = crossentropy(net(X2), y2)

	loss.backward()

	optim.step()

