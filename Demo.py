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
import one_hot_to_midi as oh
import sys

#Run a demo using the given parameters
def run_demo(num_samples, model):
	samples = None

	if model == "Note_CNN":
		weights = "NOTE_CNN_WEIGHTS_400.pt"
		samples = nrnn.generate_samples_NoteCNN(weights, 32, 10, num_samples)

	elif model == "0.01":
		weights = "Q_400-500000.pt"
		samples = DQN.generate_sample(weights, 32, 10, num_samples)

	elif model == "0.05":
		weights = "Q_500-100000.pt"
		samples = DQN.generate_sample(weights, 32, 10, num_samples)

	elif model == "0.1":
		weights = "Q-500000.pt"
		samples = DQN.generate_sample(weights, 32, 10, num_samples)

	elif model == "0.3":
		weights = "Q_300-500000.pt"
		samples = DQN.generate_sample(weights, 32, 10, num_samples)

	elif model == "0.5":
		weights = "Q_200-500000.pt"
		samples = DQN.generate_sample(weights, 32, 10, num_samples)
	else:
		print("Invalid model parameter! Try again")


	for i in range(num_samples):
		oh.one_hot_to_midi(samples[i], midi_filename='demo_song-'+str(i)+'.mid')

	return None

model = sys.argv[1]
num_samples = int(sys.argv[2])

run_demo(num_samples, model)