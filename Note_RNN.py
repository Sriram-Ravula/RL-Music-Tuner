import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

class Note_CNN(nn.Module):
    def __init__(self, batch_size, num_steps_input, n_instruments):
        super(Note_CNN, self).__init__()

        self.batch_size = batch_size

        #L_out = [L_in - kernel_size + 1] - for  L_out = L_in/1, use kernel size of (num_steps_input/2)+1
        self.convs = nn.Conv1d(in_channels=n_instruments, out_channels=64, kernel_size=int(num_steps_input/2)+1)

        self.relu1 = nn.ReLU()
        
        #flatten all the output channels into one - new size is 64*L_in/2  
        self.flatten = Flatten()

        #linearly combine input to produce 32 outputs
        self.linear = nn.Linear(in_features=64*int(num_steps_input/2), out_features=32)

        self.relu2 = nn.ReLU()

        #output layer, converts 32 values from previous layer into outputs for each category
        self.output = nn.Linear(in_features=32, out_features=n_instruments)

        self.softmax = nn.Softmax()

    def forward(self, x):
        output = self.convs(x)
        output = self.relu1(output)
        output = self.flatten(output)
        output = self.linear(output)
        output = self.relu2(output)
        output = self.output(output)
        output = self.softmax(output)

        return output.view(self.batch_size, -1)

#given an input of size (1, num_instruments, input)
def size_right(X, num_steps, num_instruments):
    snippet_length = list(X.shape)[-1] #get the number of steps given to us in the snippet

    if (num_instruments != list(X.shape)[1]): #if the number of channels do not match, fail
        return -100
    elif (num_steps < snippet_length): #if the given snippet is longer than the desired length, fail
        return -100

    output = torch.zeros(1, num_instruments, num_steps)

    output[0,:,num_steps-snippet_length:num_steps] = X

    output = Variable(output)
    output.requires_grad = False

    return output

y = Variable(torch.tensor([1,0,0,0,0,0,0,0,0,0]).view(1, 10, 1))
y.requires_grad = False

output = size_right(y, 32, 10)
print(output[0,:,0])
print(output[0,:,1])
print(output[0,:,30])
print(output[0,:,31])


X = np.zeros((32, 10))
X2 = np.zeros((32, 10))

for i in range(32):
    X[i,:] = [1,0,0,0,0,0,0,0,0,0]
    X2[i,:] = [0,0,0,0,0,0,0,0,0,1]


X = Variable(torch.tensor(X).view(1,10,32).float())
X.requires_grad = False

X2 = Variable(torch.tensor(X2).view(1,10,32).float())
X2.requires_grad = False

y = Variable(torch.tensor(np.argmax([1,0,0,0,0,0,0,0,0,0])).view(-1))
y.requires_grad = False

y2 = Variable(torch.tensor(np.argmax([0,0,0,0,0,0,0,0,0,1])).view(-1)) 
y2.requires_grad = False

net = Note_CNN(1, 32, 10)

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

print(net(X))
print(net(X2))

X3 = np.zeros((32, 10))

for i in range(32):
    X3[i,:] = [0,1,0,0,0,0,0,0,0,0]

X3 = Variable(torch.tensor(X3).view(1,10,32).float())
X3.requires_grad = False

print(net(X3))

print(list(X3.shape)[-1])