"""
https://arxiv.org/pdf/1409.3215 2014

The paper addresses the problem of sequence transduction, where an input sequence is transformed into an output sequence. 
In this paper, we show that a straightforward application of the Long Short-Term Memory (LSTM) architecture [16] can solve general sequence to sequence problems.

we reversed the order of words in the source sentence but not the target sentences in the training and test
set. 

performed good on translation tasks

We used deep LSTMs 
- with 4 layers,
- with 1000 cells at each layer and 
- 1000 dimensional word embeddings, 
- with an input vocabulary of 160,000 
- with an output vocabulary of 80,000.

Model:
LSTM encode -  four layers
LSTM decoder -  four layers
"""

from jonigrad.layers import Module, LSTM, Linear, Embedding
import numpy as np


class Encoder(Module):
    def __init__(self, input_dim, emb_dim, hid_dim):
        super().__init__()
        self.embedding = Embedding(input_dim, emb_dim)
        self.lstm = LSTM(emb_dim, hid_dim)

    def forward(self, input_seq):
        embedded = self.embedding.forward(input_seq)
        lstm_out, h, c = self.lstm.forward(embedded)
        print(h.shape, lstm_out.shape)
        return lstm_out, h, c

    def backward(self, dL_dy):
        dL_dy = np.expand_dims(dL_dy, 0)
        print(dL_dy.shape)
        dL_dy, dh, dc = self.lstm.backward(dL_dy)
        dL_dy = self.embedding.backward(dL_dy)
        return dL_dy, dh, dc

class Decoder(Module):
    def __init__(self, output_dim, emb_dim, hid_dim):
        super().__init__()
        self.embedding = Embedding(output_dim, emb_dim)
        self.lstm = LSTM(emb_dim, hid_dim)
        self.linear = Linear(hid_dim, output_dim)

    def forward(self, input_seq, hidden, cell):
        input_seq = np.expand_dims(input_seq, 1)  # (batch_size, 1)
        embedded = self.embedding.forward(input_seq)  # (batch_size, 1, emb_dim)
        lstm_out, h, c = self.lstm.forward(embedded, hidden, cell)  # lstm_out: (batch_size, 1, hid_dim)
        predictions = self.linear.forward(lstm_out[:, -1, :])  # (batch_size, output_dim)
        return predictions, h, c

    def backward(self, dL_dy):
        dL_dy = self.linear.backward(dL_dy[:, -1, :])  # (batch_size, hid_dim)
        dL_dy = np.expand_dims(dL_dy, 1)  # (batch_size, 1, hid_dim)
        dL_dy, dh, dc = self.lstm.backward(dL_dy)  # (batch_size, 1, emb_dim)
        dL_dy = self.embedding.backward(dL_dy)  # (batch_size, input_dim)
        return dL_dy, dh, dc
