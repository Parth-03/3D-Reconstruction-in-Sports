import pickle
import matplotlib.pyplot as plt
import numpy as np

num_epochs = 10
with open("data/Panda/epoch_losses.pkl", 'rb') as file:
    epoch_losses = pickle.load(file)

with open("data/Panda/batch_losses.pkl", 'rb') as file:
    batch_losses = pickle.load(file)

num_batches = len(batch_losses)

plt.plot(np.arange(num_epochs), epoch_losses)
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.title('SMPL Trans Loss Over Time')
plt.show()

plt.plot(np.arange(num_batches), batch_losses)
plt.xlabel('Batch')
plt.ylabel('Average Loss')
plt.title('SMPL Trans Loss Over Time')
plt.show()
