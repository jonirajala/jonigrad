import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from jonigrad.utils import load_fi_en_translations

# Define the Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell

# Define the Decoder
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, hidden, cell):
        trg = trg.unsqueeze(0)
        embedded = self.dropout(self.embedding(trg))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5, generator=None):
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.fc_out.out_features
        outputs = torch.zeros(trg_len, trg.shape[1], trg_vocab_size).to(self.device)
        hidden, cell = self.encoder(src)
        input = trg[0, :]
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = np.random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t, :] if teacher_force else top1
        return outputs

# Hyperparameters
ENC_EMB_DIM = 128
DEC_EMB_DIM = 128
HID_DIM = 128
N_LAYERS = 4
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
ITERS = 1000
THRESHOLD = 5
BATCH_SIZE = 32
LR = 0.01
g = torch.Generator().manual_seed(42)  # Seed it for reproducibility


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def translate_sentence(model, src_sentence, src_vocab, trg_vocab):
    model.eval()
    with torch.no_grad():
        src_tensor = torch.tensor(src_sentence, dtype=torch.long).unsqueeze(1).to(model.device)
        hidden, cell = model.encoder(src_tensor)
        trg_indexes = [trg_vocab['<SOS>']]
        for _ in range(100):  # Limit sentence length
            trg_tensor = torch.tensor([trg_indexes[-1]], dtype=torch.long).to(model.device)
            with torch.no_grad():
                output, hidden, cell = model.decoder(trg_tensor, hidden, cell)
            pred_token = output.argmax(1).item()
            trg_indexes.append(pred_token)
            if pred_token == trg_vocab['<EOS>']:
                break
    trg_tokens = [list(trg_vocab.keys())[list(trg_vocab.values()).index(i)] for i in trg_indexes]
    return trg_tokens[1:-1]  # Exclude <SOS> and <EOS> tokens

# Training loop
def train(en_data, en_vocab, fi_data, fi_vocab):
    INPUT_DIM = len(en_vocab)
    OUTPUT_DIM = len(fi_vocab)
    # Initialize encoder, decoder, and seq2seq model
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

    model = Seq2Seq(enc, dec, device=device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    model.train()
    losses = []

    pbar = tqdm(range(ITERS), desc="Training Progress")
    for i in pbar:
        ix = torch.randint(0, en_data.shape[0], (BATCH_SIZE,), generator=g)
        # Select batch data
        src, trg = torch.tensor(en_data[ix], dtype=torch.long).to(model.device), torch.tensor(fi_data[ix], dtype=torch.long).to(model.device)
        src, trg = src.reshape(-1, BATCH_SIZE), trg.reshape(-1, BATCH_SIZE)

        optimizer.zero_grad()
       
        output = model(src, trg)
       
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), THRESHOLD)
        optimizer.step()
        losses.append(loss.item())
        pbar.set_postfix({"train_loss": loss.item()})

        if i % 10 == 0:
            example_idx = torch.randint(0, en_data.shape[0], (1,)).item()
            src_sentence = en_data[example_idx]
            translation = translate_sentence(model, src_sentence, en_vocab, fi_vocab)
            print(f"\nExample translation at iteration {i}:")
            print(f"Source: {' '.join([list(en_vocab.keys())[list(en_vocab.values()).index(idx)] for idx in src_sentence]).replace('<PAD>', '')}")
            print(f"Translation: {' '.join(translation)}")

    return losses, enc, dec

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
