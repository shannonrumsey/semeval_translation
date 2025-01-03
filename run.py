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

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        """
        Feedforward class allows the option of passing in additional information about entities
        x will come in with dimensions batch_size, seq_len, n_embd
        Thus to add entities information, we will need to have a tensor of entities with shape (batch_size, seq_len, embedding dim)
        with padded and embedded entities so that each entity is the same length across each batch
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(4 * n_embd, n_embd)
        )

    def forward(self, x):

        return self.mlp(x)

class EncoderTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True, dropout=0.2)


    def forward(self, x): # no attention mask needed for encoder
        attn_output, _ = self.attention(x, x, x)
        return attn_output

class TransformerEncoder(nn.Module):
    """
    No masking for self-attention
    """
    def __init__(self, vocab_size, n_embd, n_head, n_layer, max_seq_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding = PositionalEncoding(n_embd, max_seq_len)
        self.feedforward = FeedForward(n_embd=n_embd)  # will be applied after attention blocks

        # layer normalization layers to use between attention blocks
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attention_layers = nn.ModuleList(
            [EncoderTransformerBlock(embed_dim, n_head, self.feedforward) for _ in range(n_layer)]
        )


        self.fc_out = nn.Linear(n_embd, vocab_size)

    def forward(self, x, entity_info=None):
        # entity info should be a tensor of shape (entity_seq_len, embedding dim)

        # NOTE: the forward function allows for entity information to influence the attention mechanizm in the encoder

        """
        Example:
            Input =
                (batch_size, seq_len, embedding_dim).
                Padded and encoded input sentences:
                ex "Who directed Up"
            entity info = (batch_size, entity_len, embedding_dim)
                Padded and encoded entity translations/ info
                ex. <Oben> (where oben in the german translation)

            x_with_entity = (batch_size, entity_len + seq_len, embedding_dim)
                Padded and encoded source sentence + entity translations/ info
                ex. Who directed Up <Oben>

            if the entity is present, attention will be calculated on x_with entity, then the entity will be removed
            What this looks like is the keys of the entity will be multiplied with the queries of the previous words to update the sentence
            based on the entity

            The surrounding words will also update the meaning of the entity at the end, but that part will be chopped off
            then it will go into a feedforward layer without the entity info
            then it will back into attention with the entity info added in the attention step once again then chopped
            this will repeat 
        """
        x = self.embedding(x)
        x = self.pos_embedding(x) # the possitional embedding takes in the encoded x and outputs the sum of the encoded and the poitional
        # thus x is the sum of embedding and positional as expected

        batch_size, seq_length, embed_dim = x.shape




        for layer in self.attention_layers: #each layer will be an encoder transformer block
            if entity_info is not None:
                batch_size, len_entity, embed_dim = entity_info.shape
                x_with_entity = torch.cat((x, entity_info), dim=1)
            else:
                x_with_entity = x # x is unchanged

            attention_output = layer(x_with_entity) # attention output will be (batch size, seq length, embed dim)
            # or (batch size, seq length + entity length, embed dim)

            if entity_info is not None: # REMOVE extra entity info once its been used in attention
                attention_output = attention_output[:, :-len_entity, :] # this will return our vector to (batch size, seq len, embedding dim)

            x = self.norm1(x + attention_output) # resid connection and normalize
            feedforward_output = self.feedforward(x)
            x = self.norm2(x + feedforward_output) # resid connection and normalize

        out = self.fc_out(x)

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

# in order to set the proper size for max seq length for our positional embeddings
def find_max_sequence_length(dataset):
    max_length = max(len(x_ids) for x_ids in dataset.corpus_x_ids)
    return max_length

max_seq_len_pretrain = find_max_sequence_length(pretrain_dataset)
max_seq_len_train = find_max_sequence_length(train_dataset)
max_seq_len = max(max_seq_len_pretrain, max_seq_len_train)

n_embd = 128
n_head = 4
n_layer = 6
num_epoch = 10
pos = PositionalEncoding(n_embd)
encoder = TransformerEncoder(vocab_size=vocab_size,
                             n_embd=n_embd,
                             n_head=n_head,
                             n_layer=n_layer,
                             max_seq_len= max_seq_len).to(device)
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
        


