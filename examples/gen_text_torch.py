import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

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
    input_tensor = torch.tensor(input_indices, dtype=torch.long).unsqueeze(0)
    hidden = model.init_hidden(1)

    generated_text = start_string
    for _ in range(length):
        output, hidden = model(input_tensor, hidden)
        output_idx = torch.argmax(output[0]).item()
        generated_text += idx2char[output_idx]
        input_tensor = torch.tensor([[output_idx]], dtype=torch.long)
    
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
        yield torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

# Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.embed = nn.Embedding(input_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x, hidden):
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)
        out = out.reshape(out.size(0) * out.size(1), HIDDEN_SIZE)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (weight.new(num_layers, batch_size, HIDDEN_SIZE).zero_(),
                weight.new(num_layers, batch_size, HIDDEN_SIZE).zero_())

model = LSTMModel(input_size, embed_size, HIDDEN_SIZE, num_layers, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

train_losses = []
val_losses = []

for epoch in range(num_epochs):
    hidden = model.init_hidden(BATCH_SIZE)
    epoch_train_loss = 0
    num_batches = len(train_data) // (BATCH_SIZE * seq_length)
    
    for inputs, targets in tqdm(get_batch(train_data, seq_length, BATCH_SIZE), total=num_batches, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        inputs, targets = inputs.to(torch.device('cpu')), targets.to(torch.device('cpu'))  # Ensure tensors are on the same device
        model.zero_grad()
        outputs, hidden = model(inputs, hidden)
        hidden = (hidden[0].detach(), hidden[1].detach())  # Detach hidden states to prevent backpropagating through the entire training history
        loss = criterion(outputs, targets.view(-1))
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
    
    avg_train_loss = epoch_train_loss / num_batches
    train_losses.append(avg_train_loss)

    # Validation
    model.eval()
    hidden = model.init_hidden(BATCH_SIZE)
    epoch_val_loss = 0
    num_val_batches = len(val_data) // (BATCH_SIZE * seq_length)
    
    with torch.no_grad():
        for inputs, targets in get_batch(val_data, seq_length, BATCH_SIZE):
            inputs, targets = inputs.to(torch.device('cpu')), targets.to(torch.device('cpu'))  # Ensure tensors are on the same device
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
