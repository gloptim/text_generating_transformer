import torch
import torch.nn as nn
import math

# Positional encoding to learn sequential dependency
class PositionalEncoding(nn.Module):
	def __init__(self, d_model, max_seq_length = 10000):
		super().__init__()
		pe = torch.zeros(max_seq_length, d_model)
		position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)    # Even indices: sine
		pe[:, 1::2] = torch.cos(position * div_term)    # Odd indices: cosine
		self.register_buffer('pe', pe.unsqueeze(0))     # Shape: (1, max_seq_length, d_model)
	
	def forward(self, x):
		# [batch_size, context_size, d_model]
		x = x + self.pe[:, :x.size(1)]
		return x

# Network 
# Decoder-only transformer
class MyModel(nn.Module):
	
	def __init__(self, vocab_size, context_size, d_model, d_ffn, n_heads, n_layers, padding_idx):
		super().__init__()

		print('Model created.')
		
		self.context_size = context_size

		self.token_emb = nn.Embedding(num_embeddings = vocab_size, embedding_dim = d_model, padding_idx = padding_idx)
		self.pos_enc = PositionalEncoding(d_model = d_model, max_seq_length = context_size)

		encoder_layer = nn.TransformerEncoderLayer(
			d_model=d_model,
			nhead=n_heads,
			dim_feedforward=d_ffn,
			activation='relu',
			batch_first=True
			)
		
		self.transformer_encoder = nn.TransformerEncoder(encoder_layer = encoder_layer, num_layers = n_layers)

		self.lin_out = nn.Linear(in_features = d_model, out_features = vocab_size)
		self.softmax = nn.Softmax(dim = -1)

	# In: [batch_size, context_size]
	# Out: [batch_size, context_size, vocab_size]
	def forward(self, x):
		
		x = self.token_emb(x)
		x = self.pos_enc(x)

		causal_mask = nn.Transformer.generate_square_subsequent_mask(self.context_size).to(device=x.device)
		x = self.transformer_encoder(x, mask = causal_mask)

		x = self.lin_out(x)
		x = self.softmax(x)

		return x