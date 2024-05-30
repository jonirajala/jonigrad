"""
https://arxiv.org/pdf/1706.03762
a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely
New architecture is needed because rnn's are hard to train due to sequential nature and gradients vanish
vanishing gradients causes RNN to NOT learn the long-range dependencies well across time steps


Encoder (a stack of N = 6 identical layers)
 - Layer (Each layer has twosub-layers)
    - OUTPUT: LayerNorm(x + Sublayer(x)),
    - multi-head self-attention mechanism
    - positionwise fully connected feed-forward network
 - residual connection around each of the two sub-layers followed by layer normalization
 
Decoder (a stack of N = 6 identical layers)
- In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head 
  attention over the output of the encoder stack
-  residual connections around each of the sub-layers followed by layer normalization
- We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This
  masking, combined with fact that the output embeddings are offset by one position, ensures that the
  predictions for position i can depend only on the known outputs at positions less than i.

"""