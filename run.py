from dataset import pretrain_loader, train_loader, pretrain_dataset
import sys
import torch
from torch import nn
from torch import optim
import os

seed = 27
torch.manual_seed(seed)

# Determine if GPU available
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


class PositionalEncoding(nn.Module):
    """
    Takes in an encoded input and outputs the sum of the positional representations and the learned semantic embeddings
    e.g. [The, cat, sat, on, the, mat] -> [0, 1, 2, 3, 4, 5]
    """
    def __init__(self, n_embd, max_len=5000):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, n_embd)

    def forward(self, x):
        # .view() ensures the correct tensor shape
        pos = torch.arange(x.size(1), device=x.device).view(1, x.size(1))
        embedding = self.pos_embedding(pos)
        return x + embedding
    
class TransformerEncoder(nn.Module):
    """
    No masking for self-attention
    """
    def __init__(self, vocab_size, n_embd, n_head, n_layer):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding = PositionalEncoding(n_embd)

        encoder_layer = nn.TransformerEncoderLayer(n_embd, n_head, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layer)

        self.fc_out = nn.Linear(n_embd, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_embedding(x)
        out = self.encoder(x)
        # Project output
        out = self.fc_out(out)

        return out

class TransformerDecoder(nn.Module):
    """
    Requires masking for cross-attention
    """
    def __init__(self):
        pass

    def forward(self):
        pass

vocab_size = len(pretrain_dataset.vocab)
n_embd = 128
n_head = 4
n_layer = 6
num_epoch = 10
pos = PositionalEncoding(n_embd)
encoder = TransformerEncoder(vocab_size=vocab_size,
                             n_embd=n_embd,
                             n_head=n_head,
                             n_layer=n_layer).to(device)
pad_index = pretrain_dataset.vocab["<PAD>"]
loss_fn = nn.CrossEntropyLoss(ignore_index=pad_index)
enc_optimizer = optim.AdamW(encoder.parameters(), lr=0.001)
dec_optimizer = optim.AdamW(encoder.parameters(), lr=0.001)
encoder_path = os.path.join(os.path.dirname(__file__), "encoder_model")
decoder_path = os.path.join(os.path.dirname(__file__), "decoder_model")

for step in range(num_epoch):

    encoder.train()
    # decoder.train()
    for enc, dec, trg, msk in pretrain_loader:
        enc = enc.to(device)
        dec = dec.to(device)
        trg = trg.to(device)
        msk = msk.to(device)

        enc_optimizer.zero_grad()
        dec_optimizer.zero_grad()

        encoder_outputs = encoder(enc.to(device))
        # decoder_outputs = decoder(encoder_outputs.to(device))

        enc_optimizer.step()
        dec_optimizer.step()
        


