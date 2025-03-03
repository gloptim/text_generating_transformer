import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from tgt import MyTGT

def main():

	# Open file with text
	file = open('nn/gen/text/text.txt')
	# Get text data for training
	data = file.read()
	# Select a part for learning (optional)
	data = data[32000:96000]

	# Initialize LLM
	tgt = MyTGT(data, 
		path = 'model.pt',	
		context_size = 128, 
		batch_size = 64, 
		d_model = 512, 
		n_heads = 4, 
		n_layers = 3, 
		d_ffn = 256, 
		lr = 1e-4)
	
	# Train 
	tgt.train(epochs=10, plot=True, verbose = 2, print_every = 128)
	
	# Generate and print
	text = tgt.generate(seed = 'Yes, master, tell me ', size=1024, temperature=1.1)
	print(text)

if __name__ == '__main__':

	main()
