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

#given an torch.tensor input of size (1, num_instruments, input)
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

#Given a state of shape (1, num_instruments, num_steps) and an action of size (1, num_instruments, 1), add the action to the end of the state and rotate all other actions upward
#important - given state has full 
def add_action(S, a):
	#check that the two inputs have the same number of channels
	if (list(S.shape)[1] != list(a.shape)[1]): 
		return -100
	#check that the given action has only one timestep
	if (list(a.shape)[-1] != 1):
		return -100

	num_steps = list(S.shape)[-1] #the length in timesteps of the given state 

	#will hold S rotated one row upward with a at the last row
	output = torch.empty(S.shape)

	#transfer the all but the first row of S to all but the last row of the output
	output[0,:,0:num_steps-1] = S[0,:,1:num_steps]

	#put the taken action as the last timestep
	output[0,:,num_steps-1] = a.squeeze()

	output = Variable(output)
	output.requires_grad = False

	return output


#trains the Note_CNN for a given number of iterations
def train_Note_CNN(training_data, validation_data, Note_CNN, num_epochs=1000, log_mse=False, log_every=50, filename = "NOTE_CNN_WEIGHTS.pt"):
	training_data = Variable(torch.tensor(training_data))
	validation_data = Variable(torch.tensor(validation_data))

	num_examples_validation = validation_data.shape[0]
	num_examples_train = training_data.shape[0]
	num_instruments = training_data.shape[1]
	num_steps = training_data.shape[2]

	#Make sure training and validation data have same dimension
	if (num_instruments != validation_data.shape[1] or num_steps != validation_data.shape[2]):
		return -100

	if log_mse:
		loss_log = np.array(int(num_epochs/log_every))

	crossentropy = torch.nn.CrossEntropyLoss()
	optim = torch.optim.Adam(Note_CNN.parameters(), lr=1e-3)

	for e in range(num_epochs):
		sample_order = np.random.choice(num_examples_train, num_examples_train, replace=False) #get a random ordering of samples

		for s in range(num_examples_train):
			sample = training_data[sample_order[s]].view(1, num_instruments, num_steps) #current training sample

			for i in range(7, num_steps): #starting at the eigth row, feed the traing example to the network and train
				optim.zero_grad()

				state = size_right(sample[0,:,0:i+1].view(1,num_instruments,-1), num_steps, num_instruments) #get the current state by expanding the first i rows

				net_out = Note_CNN(state) #the network output for this particular set of rows

				target = Variable(torch.argmax(sample[0,:,i+1].squeeze()).view(-1)) #take the argmax of the next row to get an index for cross entropy loss

				loss = crossentropy(net_out, target) 

				loss.backward()

				optim.step()

		#check if we must log validation MSE - if so, iterate through every example in validation set and sum loss vs net predictin
		if (log_mse and e%log_every==0): 

			for s in range(num_examples_validation):
				sample = validation_data[s].view(1, num_instruments, num_steps) #current training sample

				for i in range(7, num_steps): #starting at the eigth row, feed the traing example to the network and train
					state = size_right(sample[0,:,0:i+1].view(1,num_instruments,-1), num_steps, num_instruments) #get the current state by expanding the first i rows

					net_out = Note_CNN(state) #the network output for this particular set of rows

					target = Variable(torch.argmax(sample[0,:,i+1].squeeze()).view(-1)) #take the argmax of the next row to get an index for cross entropy loss

					loss = crossentropy(net_out, target) 

					loss_log[int(e/log_every)] += loss.detach().numpy()


	torch.save(Note_CNN.state_dict(), filename)

	
	return loss_log

	return -1
