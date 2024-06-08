import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from seq2seq import Decoder, Encoder
from jonigrad.utils import load_fi_en_translations
from jonigrad.layers import CrossEntropyLoss

BATCH_SIZE = 32
ITERS = 1000
LR = 0.1
g = np.random.default_rng()  # create a random generator
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 256
N_LAYERS = 1
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
THRESHOLD = 5
teacher_forcing_ratio = 0.5

def translate_sentence(sentence, src_vocab, trg_vocab, encoder, decoder, max_len=50):
    encoder.eval()
    decoder.eval()

    tokens = [token.lower() for token in sentence.split()]
    tokens = (
        [src_vocab["<SOS>"]]
        + [src_vocab.get(token, src_vocab["<UNK>"]) for token in tokens]
        + [src_vocab["<EOS>"]]
    )

    src_tensor = np.array(tokens).reshape(1, -1)
    _, hidden, cell = encoder.forward(src_tensor)

    trg_indexes = [trg_vocab["<SOS>"]]

    for i in range(max_len):
        trg_tensor = np.array([trg_indexes[-1]])

        output, hidden, cell = decoder.forward(trg_tensor, hidden, cell)

        pred_token = output.argmax(1)[0]
        trg_indexes.append(pred_token)

        if pred_token == trg_vocab["<EOS>"]:
            break

    trg_tokens = [
        list(trg_vocab.keys())[list(trg_vocab.values()).index(i)] for i in trg_indexes
    ]

    return trg_tokens[1:-1]


def train(en_data, en_vocab, fi_data, fi_vocab):
    INPUT_DIM = len(en_vocab)
    OUTPUT_DIM = len(fi_vocab)
    encoder = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM)
    decoder = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM)

    loss_f = CrossEntropyLoss()
    losses = []

    encoder.train()
    decoder.train()
    pbar = tqdm(range(ITERS), desc="Training Progress")
    for i in pbar:
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
        print(output.shape, trg.shape)
        loss = loss_f(output, trg)
        dL_dy = loss_f.backward()

        dL_dy = dL_dy.reshape(batch_size, trg_len - 1, trg_vocab_size)
        dL_dy_decoder = dL_dy

        dL_dy_decoder, dh, dc = decoder.backward(dL_dy_decoder)
        dL_dy_encoder = encoder.backward(dL_dy_decoder, dh, dc)

        decoder.clip_grad(THRESHOLD, BATCH_SIZE)
        encoder.clip_grad(THRESHOLD, BATCH_SIZE)

        encoder.step(LR)
        decoder.step(LR)
        pbar.set_postfix({"train_loss": loss.item()})

        losses.append(loss.item())

        if i % 10 == 0:
            test_sentence = "I am a chicken"
            translation = translate_sentence(
                test_sentence, en_vocab, fi_vocab, encoder, decoder
            )
            print("Translation:", " ".join(translation))
            encoder.train()
            decoder.train()
    return losses, encoder, decoder


if __name__ == "__main__":
    print("Loading data")
    en_data, en_vocab, fi_data, fi_vocab = load_fi_en_translations(debug=False)
    print(
        f"Data loaded, {len(en_data)} sentences, from {len(en_vocab)} vocab to {len(fi_vocab)} vocab"
    )
    losses, encoder, decoder = train(en_data, en_vocab, fi_data, fi_vocab)

    plt.plot(losses)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.show()

    test_sentence = "I am a chicken"
    translation = translate_sentence(
        test_sentence, en_vocab, fi_vocab, encoder, decoder
    )
    print(f'Translation: {" ".join(translation)}')

