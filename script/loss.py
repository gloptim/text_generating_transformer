import torch
import torch.nn as nn

class MyLoss(nn.Module):

	def __init__(self):
		super().__init__()

	def forward(self, y_true, y_pred):

		# Flatten tensors to [batch_size * seq_size, vocab_size]
		batch_size, seq_len, vocab_size = y_pred.shape
		pred_flat = y_pred.view(-1, vocab_size)
		targets_flat = y_true.view(-1)

		# Get probabilities of target indices
		probs_at_targets = pred_flat[torch.arange(targets_flat.size(0)), targets_flat]

		# Take negative log
		loss = -torch.log(probs_at_targets.clamp(min=torch.finfo(torch.float32).eps)).mean()

		return loss
