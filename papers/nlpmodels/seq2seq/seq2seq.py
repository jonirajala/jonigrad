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