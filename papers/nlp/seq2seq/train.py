import numpy as np
from tqdm import tqdm

from seq2seq import Decoder, Encoder
from jonigrad.utils import load_fi_en_translations
from jonigrad.layers import CrossEntropyLoss

BATCH_SIZE = 16
ITERS = 2
LR = 0.001
g = np.random.default_rng()  # create a random generator
# ENC_EMB_DIM = 256
# DEC_EMB_DIM = 256
# HID_DIM = 256
ENC_EMB_DIM = 16
DEC_EMB_DIM = 16
HID_DIM = 16
N_LAYERS = 1
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
teacher_forcing_ratio=0.5


def train():
    en_data, en_vocab, fi_data, fi_vocab = load_fi_en_translations(debug=True)
    
    INPUT_DIM = len(en_vocab)
    OUTPUT_DIM = len(fi_vocab)
    encoder = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM)
    decoder = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM)

    loss_f = CrossEntropyLoss()
    losses = []

    for iter in tqdm(range(ITERS)):
        ix = g.integers(low=0, high=en_data.shape[0], size=BATCH_SIZE)
        Xb, Yb = en_data[ix], fi_data[ix]

        encoder.zero_grad()
        
        trg_len = Yb.shape[1]
        batch_size = BATCH_SIZE
        trg_vocab_size = len(fi_vocab)
        
        outputs = np.zeros((batch_size, trg_len, trg_vocab_size))
        
        output, hidden, cell = encoder.forward(Xb)
        input = Yb[:, 0]
        
        for t in range(1, trg_len):
            output, hidden, cell = decoder.forward(input, hidden, cell)
            outputs[:, t] = output
            top1 = output.argmax(1)
            input = Yb[:, t] if np.random.random() < teacher_forcing_ratio else top1

        output = outputs[:, 1:].reshape(-1, trg_vocab_size)
        trg = Yb[:, 1:].reshape(-1)

        loss = loss_f(output, trg)
        dL_dy = loss_f.backward()

        dL_dy = dL_dy.reshape(batch_size, trg_len - 1, trg_vocab_size)
        # Backward pass through the decoder
        dL_dy_decoder = dL_dy
        
        dL_dy_decoder, dh, dc = decoder.backward(dL_dy_decoder)
        dh = dh.reshape(batch_size, -1)
        dc = dc.reshape(batch_size, -1)

        # Backward pass through the encoder
        dL_dy_encoder = encoder.backward(dh)

        encoder.step(LR)
        decoder.step(LR)

        losses.append(loss.item())



if __name__ == '__main__':
    train()