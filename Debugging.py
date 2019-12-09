import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import Note_RNN as nrnn
import time
import DQN 
import matplotlib.pyplot as plt

"""
print("------TESTING SIZE CORRECTION TO 32 STEPS---------")
print("Should be only last row with a 1 in the first position\n")

y = Variable(torch.tensor([1,0,0,0,0,0,0,0,0,0]).view(1, 10, 1))
y.requires_grad = False

output = nrnn.size_right(y, 32, 10)
print("Row 1: ", output[0,:,0])
print("Row 2: ", output[0,:,1])
print("Row 31: ", output[0,:,30])
print("Row 32: ", output[0,:,31])


print("\n------TESTING ADDING AN ACTION TO A STATE---------")
print("SHOULD SHOW 6 CHANNELS WITH LAST TWO ROWS HAVING 1 IN FIRST POS\n")

action_added = DQN.add_action(output, y)
print("Row 1: ", action_added[0,:,0])
print("Row 2: ", action_added[0,:,1])
print("Row 3: ", action_added[0,:,2])
print("Row 30: ", action_added[0,:,29])
print("Row 31: ", action_added[0,:,30])
print("Row 32: ", action_added[0,:,31])

print("\nSHOULD SHOW 6 CHANNELS WITH LAST THREE ROWS HAVING 1 IN FIRST POS\n")

action_added = DQN.add_action(action_added, y)

print("Row 1: ", action_added[0,:,0])
print("Row 2: ", action_added[0,:,1])
print("Row 3: ", action_added[0,:,2])
print("Row 30: ", action_added[0,:,29])
print("Row 31: ", action_added[0,:,30])
print("Row 32: ", action_added[0,:,31])


print("\n------TESTING NOTE CNN TRAINING---------")
print("SHOULD PREDICT A 1 IN THE FIRST SPOT, THEN IN LAST SPOT, THEN UNDEFINED - LASTLY SHOULD PRINT 32\n")

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

net = nrnn.Note_CNN(1, 32, 10)

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

print(net(X3).detach().cpu().numpy()[0])


print("\n------TESTING NOTE CNN TRAINING FOR REALZZ!!!!!!---------")
print("===================================================================================================\n")

data = np.load("tracks_one_hot_1100.npy")
print(data.shape)

training_data = data[0:1000]
validation_data = data[1000:1100]

print(training_data.shape)
print(validation_data.shape)


Note_CNN = nrnn.Note_CNN(1, 32, 10)
loss_log = nrnn.train_Note_CNN(training_data, validation_data, Note_CNN, num_epochs=100, log_every=10, log_loss=True, debug=True, CUDA=torch.cuda.is_available(), filename = "NOTE_CNN_WEIGHTS_400.pt")
np.save("DQN_weights/reward_log_NOTECNN_400", loss_log)


"""
print("\n------TESTING DQN iINITIALISATIONS--------")
print("===================================================================================================\n")

Q = DQN.init_DQN(32, 10, "NOTE_CNN_WEIGHTS_400.pt", CUDA=torch.cuda.is_available())
Target_Q  = DQN.init_DQN(32, 10, "NOTE_CNN_WEIGHTS_400.pt", CUDA=torch.cuda.is_available())

#DQN.update_target_DQN(Target_Q, Q, 1)

#print(Q.convs.weight)
#print(Target_Q.convs.weight)

#rando = DQN.random_one_hot(10, CUDA=torch.cuda.is_available())

#print (rando) 


print("\n------TESTING RL TRAINING--------")
print("===================================================================================================\n")

Note_CNN = nrnn.Note_CNN(1, 32, 10)
Note_CNN.load_state_dict(torch.load("NOTE_CNN_WEIGHTS_400.pt"))
Note_CNN.cuda()

rewards = DQN.Q_learning(Note_CNN, Q, Target_Q, 32, 10, "DQN_weights/Q_500", "DQN_weights/Target_Q_500", num_saves=10, num_iterations=100000, update_target_every=10, num_rewards_avg=50, CUDA = torch.cuda.is_available(), epsilon = 0.05)
np.save("DQN_weights/reward_log_DQN_500", rewards)


"""
a = np.load("DQN_weights/reward_log_DQN.npy")
b = np.load("DQN_weights/reward_log_DQN_200.npy")
c = np.load("DQN_weights/reward_log_DQN_300.npy")
d = np.load("DQN_weights/reward_log_DQN_400.npy")

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

a = running_mean(a, 1000)
b = running_mean(b, 1000)
c = running_mean(c, 1000)
d = running_mean(d, 1000)

x = range(len(list(a.squeeze())))

plt.figure()
plt.plot(x, d, label=r"$\epsilon=0.01$")
plt.plot(x, a, label=r"$\epsilon=0.1$")
plt.plot(x, c, label=r"$\epsilon=0.3$")
plt.plot(x, b, label=r"$\epsilon=0.5$")
plt.grid()
plt.ylabel("Rolling Average of 1000 Rewards")
plt.xlabel("Iteration")
plt.legend()
plt.show()

#np.save("0.5_samples", DQN.generate_sample("DQN_weights/Q_200-500000.pt", 32, 10, 100))
"""