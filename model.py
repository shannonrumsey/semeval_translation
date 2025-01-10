import torch
from torch import nn
from attention import EncoderLayers, DecoderLayers, CrossAttentionBlock

seed = 27
torch.manual_seed(seed)

# Determine if GPU available
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


class EntityEmbedding(nn.Module):
    """
    Takes in encoded entity tokens and returns entity embeddings
    Entity_tok (Tensor): BPE encoded entities
        e.g. ["Be", "yon", "ce"]
    Notes:
        - Assumes that the entities line up with the model inputs
    """
    def __init__(self, n_embd, entity_len=5000):
        super().__init__()
        self.entity_embedding = nn.Embedding(entity_len, n_embd)

    def forward(self, entity_tok):
        embedding = self.entity_embedding(entity_tok)
        return embedding


class PositionalEmbedding(nn.Module):
    """
    Takes in an embedded input and outputs the sum of the positional representations and the learned semantic embeddings
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
    def __init__(self, vocab_size, n_embd, n_head, n_layer, max_seq_len, entity_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding = PositionalEmbedding(n_embd, max_seq_len)
        self.entity_embeddings = EntityEmbedding(n_embd, entity_len)

        self.attention_layers = nn.ModuleList(
            [CrossAttentionBlock(n_embd, n_head) for _ in range(n_layer)]
        )

    def forward(self, x, entity_info=None):
        """
        Args:
            x (batch size, seq_len): Input sentences
            entity_info (batch size, entity len): Encoded entity embeddings
        Returns:
            Hidden states from Encoder
        Notes: 
            - This code implies that we will need to write a seperate function for generating embeddings for the entity information
            - Entity info should have shape (entity_seq_len, embedding dim)
        """
        x = self.embedding(x)
        x = self.pos_embedding(x)
        entity_embeddings = self.entity_embeddings(entity_info) if entity_info else None
        
        # Mimics PyTorch's TransformerEncoder
        for attention_layer in self.attention_layers: 
            # if entity info is provided, it will do cross attention using x + entity info as the keys and values and x as the query
            # if no info is provided, it will just do normal self attention
            x = attention_layer(x, x, entity_embeddings=None)

        return x


class TransformerDecoder(nn.Module):
    """
    Requires masking for self-attention
    """
    def __init__(self, max_seq_length, n_embd, n_head, vocab_size, entity_len, attention_layers=5):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)  # will give us token embeddings
        self.position_embedding_table = nn.Embedding(max_seq_length,
                                                     n_embd)
        self.last_linear_layer = nn.Linear(n_embd, vocab_size)
        self.entity_embeddings = EntityEmbedding(n_embd, entity_len)

        self.self_attention_layers = nn.ModuleList(
            [DecoderLayers(n_embd, n_head) for _ in range(attention_layers)]
        )
        self.cross_attention_layers= nn.ModuleList(
            [CrossAttentionBlock(n_embd, n_head) for _ in range(attention_layers)]
        )

    def forward(self, decoder_input, encoder_output, entity_info=None):
        """
        Uses cross-attention and self-attention
        Args:
            encoder_input (batch_size, decoder_seq_len): Input to Encoder
            decoder_input (batch size, seq_len): Input to Decoder
            entity_info (entity_seq_len, embedding dim): Entity embeddings
        Returns:
            Translated Sentence (batch_size, seq_len, vocab_size)
        Notes: 
            - This code implies that we will need to write a seperate function for generating embeddings for the entity information
            - Entity info should have shape (entity_seq_len, embedding dim)
        """
        seq_len = decoder_input.size(1)

        # embedd the decoder input 
        token_embeddings = self.token_embedding_table(decoder_input)  # batch, seq len, embedding size

        position_embeddings = self.position_embedding_table(torch.arange(seq_len, device=device))
        
        x = token_embeddings + position_embeddings  # adding token and position embeddings

        entity_embeddings = self.entity_embeddings(entity_info) if entity_info else None

        # pass through self-attention layers
        for self_attn_block in self.self_attention_layers:
            x = self_attn_block(x, entity_embeddings)

        # pass through cross-attention layers
        for cross_attn_block in self.cross_attention_layers:
            x = cross_attn_block(x, encoder_output, entity_embeddings)

        # final linear layer to map to vocab size
        output = self.last_linear_layer(x)  # (batch_size, seq_len, vocab_size)

        return output
