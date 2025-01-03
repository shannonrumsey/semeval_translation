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




from Attention import EncoderAttentionBlock # see the file Attention.py
from Attention import DecoderAttentionBlock # see the file Attention.py
from Attention import CrossAttentionBlock # see the file Attention.py

class TransformerEncoder(nn.Module):
    """
    No masking for self-attention
    """
    def __init__(self, vocab_size, n_embd, n_head, n_layer, max_seq_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding = PositionalEncoding(n_embd, max_seq_len)

        self.attention_layers = nn.ModuleList(
            [EncoderAttentionBlock(embed_dim, n_head) for _ in range(n_layer)]
        )


    def forward(self, x, entity_info = None):
        """
        :param x: dim =  (batch size, seq_len)
        :param embedded entity_info: dim = (entity_seq_len, embedding dim)
        :return: encoding embedding
        NOTE: this code implies that we will need to write a seperate function for generating embeddings for the entity information
        """
        # entity info should be a tensor of shape (entity_seq_len, embedding dim)

        x = self.embedding(x)
        x = self.pos_embedding(x) # the possitional embedding takes in the encoded x and outputs the sum of the encoded and the poitional
        # thus x is the sum of embedding and positional as expected



        for attention_layer in self.attention_layers: #each layer will be a encoder attention block
            x = attention_layer(x, entity_info = None) # this can be changed if entity info is present
            # Once again, see Attention.py for the documentation of this function,
            # but esentially it mimics a full attention block with attention, feedforward, residual connections and layer normalization
            # it uses dropout = .2 in attention and dropout = .3 in feedforward

        # no linear layer in the output is needed

        return x

class TransformerDecoder(nn.Module):
    """
    Requires masking for cross-attention
    """


    def __init__(self, max_seq_length, embed_dim, num_heads, vocab_size, attention_layers=5):
        super(DecoderTransformerTranslator, self).__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embed_dim)  # will give us token embeddings
        self.position_embedding_table = nn.Embedding(max_seq_length,
                                                     embed_dim)
        self.last_linear_layer = nn.Linear(embed_dim,
                                           vocab_size)  # go from embeddings to outputs. Made sure that this is the vocab size in order to not have errors where the perplexity is artifically low
        # normalization for input


        self.self_attention_layers = nn.ModuleList(
            [DecoderAttentionBlock(embed_dim, num_heads, self.feedforward) for _ in range(attention_layers)]
        ) # see Attention.py for documentation
        self.cross_attention_layers= nn.ModuleList(
            [CrossAttentionBlock(embed_dim, num_heads, self.feedforward) for _ in range(attention_layers)]
        ) # see Attention.py for documentation

    def forward(self, decoder_input, encoder_output, entity_info = None):
        """
        Forward pass through the decoder, using both self-attention and cross-attention.

        :param decoder_input: The input to the decoder (batch_size, decoder_seq_len)
        :param encoder_output: The output from the encoder (batch_size, encoder_seq_len, embed_dim)
        :param embedded entity_info: dim = (entity_seq_len, embedding dim)
        :return: The decoder output (batch_size, seq_len, vocab_size)


        """
        seq_len = decoder_input.size(1)

        # embedd the decoder input 
        token_embeddings = self.token_embedding_table(decoder_input)  # batch, seq len, embedding size

        position_embeddings = self.position_embedding_table(torch.arange(seq_length, device=device))
        
        x = token_embeddings + position_embeddings  # adding token and position embeddings

        # pass through self-attention layers
        for self_attn_block in self.self_attention_layers:
            x = self_attn_block(x, entity_info)

        # pass through cross-attention layers
        for cross_attn_block in self.cross_attention_layers:
            x = cross_attn_block(x, encoder_output, entity_info)

        # final linear layer to map to vocab size
        output = self.last_linear_layer(x)  # (batch_size, seq_len, vocab_size)

        return output

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
decoder = TransformerDecoder(max_seq_length = max_seq_len, embed_dim = n_embd, num_heads = n_head, vocab_size = vocab_size).to(device)
pad_index = pretrain_dataset.vocab["<PAD>"]
loss_fn = nn.CrossEntropyLoss(ignore_index=pad_index)
enc_optimizer = optim.AdamW(encoder.parameters(), lr=0.001)
dec_optimizer = optim.AdamW(encoder.parameters(), lr=0.001)
encoder_path = os.path.join(os.path.dirname(__file__), "encoder_model")
decoder_path = os.path.join(os.path.dirname(__file__), "decoder_model")

for step in range(num_epoch):

    encoder.train()
    decoder.train()

    for enc, dec, trg, msk in pretrain_loader:
        enc = enc.to(device)
        dec = dec.to(device)
        trg = trg.to(device)
        msk = msk.to(device)

        enc_optimizer.zero_grad()
        dec_optimizer.zero_grad()


        encoder_outputs = encoder(enc)


        decoder_outputs = decoder(dec, encoder_outputs)


        loss = loss_fn(decoder_outputs.view(-1, vocab_size), trg.view(-1))


        loss.backward()

        enc_optimizer.step()
        dec_optimizer.step()

        print(f"Epoch {step+1}/{num_epoch}, Loss: {loss.item()}")

        


