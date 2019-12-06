import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
import Note_RNN

class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

class DQN(nn.Module):
    def __init__(self, num_steps_input, n_instruments):
        super(DQN, self).__init__()

        #L_out = [L_in - kernel_size + 1] - for  L_out = L_in/1, use kernel size of (num_steps_input/2)+1
        self.convs = nn.Conv1d(in_channels=n_instruments, out_channels=64, kernel_size=int(num_steps_input/2)+1)

        self.relu1 = nn.ReLU()
        
        #flatten all the output channels into one - new size is 64*L_in/2  
        self.flatten = Flatten()

        #linearly combine input to produce 32 outputs
        self.linear = nn.Linear(in_features=64*int(num_steps_input/2), out_features=32)

        self.relu2 = nn.ReLU()

        #output layer, converts 32 values from previous layer into outputs for each category
        self.output = nn.Linear(in_features=32, out_features=1)

    def forward(self, x):
        output = self.convs(x)
        output = self.relu1(output)
        output = self.flatten(output)
        output = self.linear(output)
        output = self.relu2(output)
        output = self.output(output)

        return output.view(1)

#Initialise Deep Q network with weights from a trained Note CNN
def init_DQN(num_steps_input, n_instruments, filename, CUDA=False):
	initial_weights = torch.load(filename)

	net = DQN(num_steps_input, n_instruments) #initialise the Deep Q network

	net.convs.weight.data = initial_weights["convs.weight"].clone()
	net.convs.bias.data = initial_weights["convs.bias"].clone()

	net.linear.weight.data = initial_weights["linear.weight"].clone()
	net.linear.bias.data = initial_weights["linear.bias"].clone()
	
	#net.output.weight.data = initial_weights["output.weight"].clone()
	#net.output.bias.data = initial_weights["output.bias"].clone()

	if CUDA:
		net.cuda()

	return net

#Linearly anneals and updates weights of target DQN
def update_target_DQN(Target_DQN, DQN, eta):

	Target_DQN.convs.weight.data = (1-eta)*Target_DQN.convs.weight.data + eta*DQN.convs.weight.data
	Target_DQN.convs.bias.data = (1-eta)*Target_DQN.convs.bias.data + eta*DQN.convs.bias.data

	Target_DQN.linear.weight.data = (1-eta)*Target_DQN.linear.weight.data + eta*DQN.linear.weight.data
	Target_DQN.linear.bias.data = (1-eta)*Target_DQN.linear.bias.data + eta*DQN.linear.bias.data
	
	Target_DQN.output.weight.data = (1-eta)*Target_DQN.output.weight.data + eta*DQN.output.weight.data
	Target_DQN.output.bias.data = (1-eta)*Target_DQN.output.bias.data + eta*DQN.output.bias.data


#returns a random one-hot vector of shape (1, num_instruments, 1)
def random_one_hot(num_instruments, CUDA=False):
	dtype = torch.FloatTensor
	if CUDA:
		dtype = torch.cuda.FloatTensor

	index = np.random.choice(num_instruments) #get a random index to make one-hot

	output = torch.zeros(num_instruments).type(dtype)
	output[index] = 1.0

	return output.view(1, -1, 1), index

#given an index, returns a one-hot vector of shape (1, num_instruments, 1)
def one_hot(index, num_instruments, CUDA=False):
	dtype = torch.FloatTensor
	if CUDA:
		dtype = torch.cuda.FloatTensor

	output = torch.zeros(num_instruments).type(dtype)
	output[index] = 1.0

	return output.view(1, -1, 1)

#Epsilon greedy action selection 
def take_greedy_action(epsilon):
	p = random.random()

	if (p < epsilon):
		return False
	else:
		return True

#Given a state of shape (1, num_instruments, num_steps) and an action of size (1, num_instruments, 1), add the action to the end of the state and rotate all other actions upward
#important - given state has full 
def add_action(S, a, CUDA=False):
	#check that the two inputs have the same number of channels
	if (list(S.shape)[1] != list(a.shape)[1]): 
		return -100
	#check that the given action has only one timestep
	if (list(a.shape)[-1] != 1):
		return -100

	num_steps = list(S.shape)[-1] #the length in timesteps of the given state 

	#will hold S rotated one row upward with a at the last row
	if CUDA:
		output = torch.empty(S.shape).type(torch.cuda.FloatTensor)
	else:
		output = torch.empty(S.shape)

	#transfer the all but the first row of S to all but the last row of the output
	output[0,:,0:num_steps-1] = S[0,:,1:num_steps].clone()

	#put the taken action as the last timestep
	output[0,:,num_steps-1] = a.squeeze().clone()

	output = Variable(output)
	output.requires_grad = False

	return output

#Given a Q-network, state, and number of channels - returns the action that maximizes value as (index of action, action vector)
def find_best_action(Q, S, num_instruments, CUDA=False):
	a_best = 0 #argmax{a} Q(s,a,theta)
	max_val = float('-inf')

	#iterate through actions to find the best one for current time step
	for index in range(num_instruments):
		candidate_a = add_action(S, one_hot(index, num_instruments, CUDA), CUDA) #the candidate action

		Q_candidate_a = Q(candidate_a).detach().cpu().numpy()[0]

		if (Q_candidate_a > max_val): #if Q(s,candidate_a,theta) is better than the last candidate, it is optimal
			a_best = index
			max_val = Q_candidate_a

	return a_best, max_val


#DQN_filename and Target_DQN_filename give the file prefix of the location to intermittently save weights - will append a "-#.pt" to each
#num_saves - the number of times to save the DQN and Target DQN weights
#update_target_every - the number of iterations to update the target DQN weights
#num_rewards_avg - the number of rewards to maintain a rolling average of
#epsilon - for e-greedy action selection
#eta - the annealing factor for Target update
#gamma - discount factor 
def Q_learning(Note_CNN, DQN, Target_DQN, num_steps, num_instruments, DQN_filename, Target_DQN_filename, num_saves = 10, num_iterations=10000, update_target_every=10, num_rewards_avg = 50, epsilon = 0.1, eta = 0.01, gamma=0.5, CUDA=False):
	dtype = torch.FloatTensor
	if CUDA:
		dtype = torch.cuda.FloatTensor

	rewards_log = np.zeros(num_rewards_avg) #will hold last r rewards
	rolling_avg_rewards = np.zeros(num_iterations) #will hold the rolling average of last r rewards

	save_every = int(num_iterations/num_saves) #save model weights every n iterations

	#choose a random vector to use as S0
	S = random_one_hot(num_instruments, CUDA=CUDA)[0]
	S = Note_RNN.size_right(S, num_steps, num_instruments, CUDA=CUDA)

	optim = torch.optim.Adam(DQN.parameters(), lr=1e-3)
	mseloss = torch.nn.MSELoss(reduction='sum')
	if CUDA:
		mseloss = mseloss.cuda()

	start = time.time()
	for i in range(num_iterations):
		a_best = find_best_action(DQN, S, num_instruments, CUDA)[0] #get the index of the best action for current state

		#epsilon-greedy action selection
		if (not take_greedy_action(epsilon)):
			a, a_best = random_one_hot(num_instruments, CUDA)
		else:
			a = one_hot(a_best, num_instruments, CUDA)

		S_next = add_action(S, a, CUDA) #get S'
		Q_cur = DQN(S_next) #Q(s,a,theta)


		p_a_s = Note_CNN(S).detach().cpu().numpy()[0][a_best] #p(a|s)
		if(p_a_s == 0):
			p_a_s = 1e-8 #to prevent divide by zero

		reward = np.log(p_a_s) #the reward for taking this action 
		rewards_log[i%num_rewards_avg] = reward #store the reward in the log
		rolling_avg_rewards[i] = np.mean(rewards_log) #store the rolling average reward


		target_val = find_best_action(Target_DQN, S_next, num_instruments, CUDA)[1] #get the target DQNs best action-value for next state


		optim.zero_grad()

		loss = mseloss(Variable(torch.tensor(reward).type(dtype).view(1), requires_grad=False) + Variable(torch.tensor(gamma*target_val).type(dtype).view(1), requires_grad=False), Q_cur)

		loss.backward()

		optim.step()


		S = S_next #update state

		#check if we should save the DQN and target DQN
		if((i+1)%save_every == 0):
			torch.save(DQN.cpu().state_dict(), DQN_filename+"-"+str(i+1)+".pt")
			torch.save(Target_DQN.cpu().state_dict(), Target_DQN_filename+"-"+str(i+1)+".pt")

			if(CUDA):
				DQN.cuda()
				Target_DQN.cuda()

			print("Iteration ", i)
			print("Time: ", time.time()-start)
			print("\n")		

		#Check if we should update Target_DQN weights
		if(i%update_target_every == 0):
			update_target_DQN(Target_DQN, DQN, eta)


	return rolling_avg_rewards