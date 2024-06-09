

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

from jonigrad.layers import *


# Hyperparameters
embed_size = 256
HIDDEN_SIZE = 256
num_layers = 4
num_epochs = 20
LR = 0.001
seq_length = 30
BATCH_SIZE = 32

corpus = ""

dataset = load_dataset("opus_books", "en-fi")["train"]
for example in dataset:
    corpus += example["translation"]["fi"]

dataset = load_dataset("EkBass/fin-eng-dataset")["train"]
for i in range(len(dataset)):
    corpus += dataset[i]["Finnish"]

chars = sorted(list(set(corpus)))
char2idx = {c: i for i, c in enumerate(chars)}
idx2char = {i: c for i, c in enumerate(chars)}

input_size = len(chars)
num_classes = len(chars)

corpus_indices = [char2idx[c] for c in corpus]

split_idx = int(0.8 * len(corpus_indices))
train_data = corpus_indices[:split_idx]
val_data = corpus_indices[split_idx:]

# Text generation
def generate_text(model, start_string, length):
    model.eval()
    input_indices = [char2idx[c] for c in start_string]
    input_tensor = np.expand_dims(np.array(input_indices, dtype=np.longlong), 0)
    hidden = model.init_hidden(1)

    generated_text = start_string
    for _ in range(length):
        output, hidden = model(input_tensor, hidden)
        output_idx = np.argmax(output[0]).item()
        generated_text += idx2char[output_idx]
        input_tensor = np.array([[output_idx]], dtype=np.longlong)
    
    return generated_text

def get_batch(data, seq_length, batch_size):
    num_batches = len(data) // (batch_size * seq_length)
    data = data[:num_batches * batch_size * seq_length]
    data = np.reshape(data, (batch_size, -1))
    for i in range(0, data.shape[1], seq_length):
        x = data[:, i:i+seq_length]
        y = np.copy(x)
        if i + seq_length < data.shape[1]:
            y[:, :-1], y[:, -1] = x[:, 1:], data[:, i + seq_length]
        else:
            y[:, :-1], y[:, -1] = x[:, 1:], data[:, 0]
        yield np.array(x, dtype=np.longlong), np.array(y, dtype=np.longlong)

# Model
class LSTMModel(Module):
    def __init__(self, input_size, embed_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.embed = Embedding(input_size, embed_size)
        self.lstm = LSTM(embed_size, hidden_size, num_layers)
        self.fc = Linear(hidden_size, num_classes)
    
    def forward(self, x, h0, c0):
        x = self.embed(x)
        out, h0, c0 = self.lstm(x, h0, c0)
        out = out.reshape(out.size(0) * out.size(1), HIDDEN_SIZE)
        out = self.fc(out)
        return out, h0, c0
    
    def backward(self, dL_dy):
        dL_dx = self.fc.backward(dL_dy)
        out = out.reshape(out.size(0) * out.size(1), HIDDEN_SIZE)
        dL_dx = self.lstm.backward(dL_dx)
        _ = self.embed.backward(dL_dx)
        return dL_dx

    def init_hidden(self, batch_size):
        return self.lstm.init_hidden(batch_size)

model = LSTMModel(input_size, embed_size, HIDDEN_SIZE, num_layers, num_classes)
criterion = CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=LR)

train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    h0, c0 = model.init_hidden(BATCH_SIZE)
    epoch_train_loss = 0
    num_batches = len(train_data) // (BATCH_SIZE * seq_length)
    
    for inputs, targets in tqdm(get_batch(train_data, seq_length, BATCH_SIZE), total=num_batches, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        model.zero_grad()
        outputs, h0, c0 = model(inputs, h0, c0)
        # hidden = (hidden[0].detach(), hidden[1].detach())  # Detach hidden states to prevent backpropagating through the entire training history
        loss = criterion(outputs, targets.view(-1))
        dL_dy = criterion.backward()
        model.backward(dL_dy)
        model.step(LR)
        epoch_train_loss += loss.item()
    
    avg_train_loss = epoch_train_loss / num_batches
    train_losses.append(avg_train_loss)

    # Validation
    model.eval()
    hidden = model.init_hidden(BATCH_SIZE)
    epoch_val_loss = 0
    num_val_batches = len(val_data) // (BATCH_SIZE * seq_length)
    
    for inputs, targets in get_batch(val_data, seq_length, BATCH_SIZE):
        outputs, hidden = model(inputs, hidden)
        hidden = (hidden[0].detach(), hidden[1].detach())  # Detach hidden states
        loss = criterion(outputs, targets.view(-1))
        epoch_val_loss += loss.item()
    
    avg_val_loss = epoch_val_loss / num_val_batches
    val_losses.append(avg_val_loss)
    
    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    start_string = "Me olemme"
    generated_text = generate_text(model, start_string, 100)
    print(generated_text)

plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
plt.title('Training and Validation Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()