import os
import random
import numpy as np

import string
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch import distributions

from torchinfo import summary

import matplotlib.pyplot as plt

from tokenizer import MyTokenizer
from network import MyModel
from loss import MyLoss

device = None

# TGT - Text Generating Transformer 
# 1) Load data or model
# 2) Train model 
# 3) Generate text
class MyTGT():

	def __init__(self, data = None, path = None, context_size = 128, batch_size = 64, d_model = 512, n_heads = 4, n_layers = 3, d_ffn = 512, lr = 3e-4):

		print('-'*96)
		
		print('TGT Initialization.')

		# Select CUDA if available
		global device
		if not device:
			device = 'cuda' if torch.cuda.is_available() else 'cpu'

		print(f'Selected device {device}.')

		# Split string data on symbols
		self.data = copy.deepcopy(data) # Avoid shallow copy
		self.data = list(data)
		data = self.filter_data(data) # Filter garbage

		# Convert data to numpy and get its size
		self.data = np.array(self.data)
		self.data_size = len(self.data)

		print(f'Loaded {self.data_size} tokens.')
		print(f'Example: {self.data[-128:]}')

		# Create tokenizer and get vocab
		self.tokenizer = MyTokenizer(context_size = context_size, data = self.data)
		self.vocab = self.tokenizer.vocab

		print(f'Vocabulary size {len(self.vocab)} characters.')
		print(f'Vocabulary: {self.vocab}')
		
		# Encode data
		self.data = self.tokenizer.encode_all(self.data)

		# Dataset params
		self.context_size = context_size
		self.batch_size = batch_size
		self.dataset_size = 0
		
		# Prepare x/y pairs
		self.xs, self.ys, self.data_size = self.get_dataset()
		
		print(f'Sample X = {self.tokenizer.decode_all(self.xs[0])}')
		print(f'Sample Y = {self.tokenizer.decode_all(self.ys[0])}')
		
		# Split pairs on batches
		xs_batches, ys_batches, self.dataset_size = self.get_batches()
		print(f'Splitted on {self.dataset_size} batches.')
		
		# Get tensors of batches
		self.xs_batch_tensor = torch.tensor(xs_batches, dtype = torch.int32).to(device=torch.device(device))
		self.ys_batch_tensor = torch.tensor(ys_batches, dtype = torch.int32).to(device=torch.device(device))

		# Create MOC
		self.model = MyModel(vocab_size = len(self.vocab),
			context_size = context_size, 
			d_model = d_model, 
			n_heads = n_heads, 
			n_layers = n_layers, 
			d_ffn = d_ffn, 
			padding_idx = self.tokenizer.padding_token).to(device=torch.device(device))
		self.optimizer = optim.Adam(params = self.model.parameters(), lr = lr)
		self.criterion = MyLoss()

		# Display info
		summary(model = self.model, input_data = self.xs_batch_tensor[0], device = device)

		# Load
		if path:
			checkpoint = torch.load(path, weights_only=True)
			self.model.load_state_dict(checkpoint['model_state_dict'])
			self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

		print('-'*96)

	# Filter from garbage
	def filter_data(self, data):
		return [c for c in data if (c in string.ascii_letters) or (c in string.digits) or (c in string.punctuation) or (c in {' ', '\t', '\n'})]

	# Separates sequence on subsequences and related subsequences with intersections.
	def get_dataset(self):

		x, y = [], []

		for e in range(self.data_size - self.context_size - 1):
			xv, yv = [], []
			for se in range(self.context_size):
				xv.append(self.data[e+se])
				yv.append(self.data[e+se+1])

			x.append(xv)
			y.append(yv)
	
		return x, y, self.data_size - self.context_size - 1

	# Get shuffled batches 
	def get_batches(self):

		num_batches = int(self.data_size / self.batch_size)
		xs = self.xs[-num_batches * self.batch_size:]
		ys = self.ys[-num_batches * self.batch_size:]

		x_batch, y_batch = [], []

		for s in range(num_batches):
			x_batch.append(xs[s * self.batch_size : s * self.batch_size + self.batch_size])
			y_batch.append(ys[s * self.batch_size : s * self.batch_size + self.batch_size])

		return x_batch, y_batch, num_batches
	
	# Train function
	def train_step(self, x_batch, y_batch):

		y_batch_predicted = self.model(x = x_batch)
		loss = self.criterion(y_batch, y_batch_predicted)
		
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		
		return loss.item()

	# Train loop
	def train(self, epochs = 10, plot = False, verbose = 2, print_every = 128):

		_ = input('Press ENTER to start training...')

		loss_history = []

		for e in range(epochs):
			
			loss_sum = 0
			
			# Iterate over tensors and train model
			self.model.train(True)

			for step in random.sample(list(range(self.dataset_size)), self.dataset_size):
			
				loss = self.train_step(self.xs_batch_tensor[step], self.ys_batch_tensor[step])
				loss_sum = loss_sum + loss

				if verbose >= 2:
					if not step % print_every:
						print(f'STEP: {step} | LOSS: {loss} ')

			self.model.train(False)

			# Save
			torch.save({
			'model_state_dict': self.model.state_dict(),
			'optimizer_state_dict': self.optimizer.state_dict(),
			}, 'model.pt')

			loss_avg = loss_sum / self.dataset_size

			if verbose >= 1:
				print(f'EPOCH: {e} | LOSS AVG: {loss_avg}')
				print('-'*96)

			loss_history.append(loss_avg)

			if plot:
				plt.plot([x for x in range(e+1)], loss_history)
				plt.show(block = False)
				plt.pause(0.01)
				plt.clf()

	# Generate text
	def generate(self, seed = 'Yes, master, tell me ', size = 128, temperature = 1.25):

		# Crop seed to context size
		seed = list(seed)[-self.context_size:]

		# Convert seed to tokens
		seed_token = self.tokenizer.encode_all(seed)
		result_token = copy.deepcopy(seed_token)

		# Generate tokens
		total_size = len(result_token) + size
		while len(result_token) < total_size:

			with torch.no_grad():
				
				# Convert seed and reshape
				seed_tensor = torch.tensor(seed_token, dtype=torch.int32).to(device=torch.device(device)).reshape([1, -1])
				# Padding
				seed_tensor = nn.functional.pad(seed_tensor, (0, self.context_size - seed_tensor.shape[-1]), value = self.tokenizer.padding_token)

				# Get probs, div by temperature to increase entropy
				probs = self.model(x = seed_tensor).reshape([-1, len(self.vocab)])[-1] / temperature

				# Sample token, add to seed, cropt to context size
				dist = distributions.Categorical(probs = probs)
				seed_token.append(dist.sample().item())
				seed_token = seed_token[-self.context_size:]

				# Add generated token to the result
				result_token.append(seed_token[-1])

		# Decoder result and return
		result = self.tokenizer.decode_all(result_token)
		result = ''.join(result)

		return result