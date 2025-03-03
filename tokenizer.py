import numpy as np

# Tokenizer (encoder-decoder)
class MyTokenizer():

	def __init__(self, context_size, data, padding = 'PAD', unknown = 'UNK'):

		print('Tokenizer created.')

		self.context_size = context_size
		
		# Get vocabulary
		self.vocab = np.unique(data)

		# Get special token ids
		self.padding_token = len(self.vocab)
		self.unknown_token = len(self.vocab) + 1

		# Add special tokens
		self.vocab = np.append(self.vocab, [padding])
		self.vocab = np.append(self.vocab, [unknown])
		
	def encode(self, x):
		try:
			return np.where(self.vocab == x)[0][0]
		except:
			return self.unknown_token
	
	def decode(self, x):
		try:
			return self.vocab[x]
		except:
			return self.vocab[self.unknown_token]

	def encode_all(self, x):
		enc = [self.encode(x[e]) for e in range(len(x))]
		return enc

	def decode_all(self, x):
		return [self.decode(x[e]) for e in range(len(x))]