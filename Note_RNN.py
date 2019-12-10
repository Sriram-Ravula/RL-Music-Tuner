import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
import DQN

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
def size_right(X, num_steps, num_instruments, CUDA=False):
    snippet_length = list(X.shape)[-1] #get the number of steps given to us in the snippet

    if (num_instruments != list(X.shape)[1]): #if the number of channels do not match, fail
        return -100
    elif (num_steps < snippet_length): #if the given snippet is longer than the desired length, fail
        return -100

    if CUDA:
    	output = torch.zeros(1, num_instruments, num_steps).type(torch.cuda.FloatTensor)
    else:
    	output = torch.zeros(1, num_instruments, num_steps)

    output[0,:,num_steps-snippet_length:num_steps] = X.clone()

    output = Variable(output)
    output.requires_grad = False

    return output

#Generate Samples from a NoteCNN
def generate_samples_NoteCNN(weight_filename, num_steps, num_instruments, num_samples = 1):
	CNN = Note_CNN(1, num_steps, num_instruments) #initialise a Note CNN to produce samples with
	CNN.load_state_dict(torch.load(weight_filename)) 

	sample_list = np.empty((num_samples, num_instruments, num_steps)) #the list of samples to return

	#Iterate through the number of required samples
	for n in range(num_samples):
		S = DQN.random_one_hot(num_instruments)[0] #get a random seed and correct its size
		S = size_right(S, num_steps, num_instruments)

		#Iterate through the last n-1 places and use the probability distribution of the Note CNN to pick actions
		for t in range(num_steps-1): 
			action_probs = CNN(S).detach().cpu().numpy()[0].squeeze() #get the probability distribuition p(a|s)

			a_index = np.random.choice(a=num_instruments, p=action_probs) #choose a random action based on p(a|s)

			a = DQN.one_hot(a_index, num_instruments) #the one-hot vector corresponsing to a_best

			S_next = DQN.add_action(S, a) #the next state after taking the greedy action

			S = S_next #the next state is now the current state

		sample_list[n] = S.detach().cpu().numpy()[0] #add the sample to the list to be output

	return sample_list



#trains the Note_CNN for a given number of iterations
def train_Note_CNN(training_data, validation_data, Note_CNN, num_epochs=1000, log_loss=False, log_every=50, filename = "NOTE_CNN_WEIGHTS.pt", debug=False, CUDA = False):
	dtype = torch.FloatTensor

	if CUDA:
		Note_CNN = Note_CNN.cuda()
		dtype = torch.cuda.FloatTensor

	training_data = Variable(torch.tensor(training_data).type(dtype))
	training_data.requires_grad = False
	validation_data = Variable(torch.tensor(validation_data).type(dtype))
	validation_data.requires_grad = False

	num_examples_validation = validation_data.shape[0]
	num_examples_train = training_data.shape[0]
	num_instruments = training_data.shape[1]
	num_steps = training_data.shape[2]

	#Make sure training and validation data have same dimension
	if (num_instruments != validation_data.shape[1] or num_steps != validation_data.shape[2]):
		return -100

	if log_loss:
		loss_log = np.zeros(int(num_epochs/log_every))

	crossentropy = torch.nn.CrossEntropyLoss()
	if CUDA:
		crossentropy = crossentropy.cuda()

	optim = torch.optim.Adam(Note_CNN.parameters(), lr=1e-3)

	start = time.time()

	for e in range(num_epochs):
		sample_order = np.random.choice(num_examples_train, num_examples_train, replace=False) #get a random ordering of samples

		for s in range(num_examples_train):
			sample = training_data[sample_order[s]].view(1, num_instruments, num_steps) #current training sample

			for i in range(8, num_steps): #starting at the eigth row, feed the traing example to the network and train
				optim.zero_grad()

				state = size_right(sample[0,:,0:i].view(1,num_instruments,-1), num_steps, num_instruments, CUDA=CUDA) #get the current state by expanding the first i rows

				net_out = Note_CNN(state) #the network output for this particular set of rows

				target = Variable(torch.argmax(sample[0,:,i].squeeze()).view(-1)) #take the argmax of the next row to get an index for cross entropy loss

				loss = crossentropy(net_out, target) 

				loss.backward()

				optim.step()

		#check if we must log validation MSE - if so, iterate through every example in validation set and sum loss vs net predictin
		if (log_loss and e%log_every==0): 

			for s in range(num_examples_validation):
				sample = validation_data[s].view(1, num_instruments, num_steps) #current training sample

				for i in range(8, num_steps): #starting at the eigth row, feed the traing example to the network and train
					state = size_right(sample[0,:,0:i].view(1,num_instruments,-1), num_steps, num_instruments, CUDA=CUDA) #get the current state by expanding the first i rows

					net_out = Note_CNN(state) #the network output for this particular set of rows

					target = Variable(torch.argmax(sample[0,:,i].squeeze()).view(-1)) #take the argmax of the next row to get an index for cross entropy loss

					loss = crossentropy(net_out, target) 

					loss_log[int(e/log_every)] += loss.detach().cpu().numpy()
			
			if debug:
				print("Iteration ", e)
				print("Time: ", time.time()-start)
				print(loss_log[int(e/log_every)])
				print("\n")


	torch.save(Note_CNN.cpu().state_dict(), filename)

	if log_loss:
		return loss_log

	return -1
