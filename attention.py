import torch
import torch.nn as nn
import sys
from dataset import semeval_train_dataset

seed = 27
torch.manual_seed(seed)

# Determine if GPU available
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


'''
This file contains code for the decoder self attention, encoder self attention, and cross attention

All attention functions allow the user to pass in additional entity info. This is how that info is incorporated:

        Example:
            Input =
                (batch_size, seq_len, embedding_dim).
                Padded and encoded input sentences:
                ex "Who directed Up"
            entity embeddings = (batch_size, entity_len, embedding_dim)
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

'''

class EncoderLayers(nn.Module):
    def __init__(self, n_embd, n_head):
        """
        Args:
            n_embd: The embedding dimension
            n_head: The number of attention heads
            x: Encoder input sentence
            entity_embeddings (Optional): Entity embeddings

        Function preforms unmasked self attention, complete with a feedforward layer, layer normalization, and residual connections
        Function gives option to include entity embeddings for experimentation
        """
        super().__init__()
        self.EncoderAttention = nn.MultiheadAttention(n_embd, n_head, batch_first=True, dropout=0.1)

        # for self attention in encoder
        self.Enorm1 = nn.LayerNorm(n_embd)
        self.Enorm2 = nn.LayerNorm(n_embd)

        # feedforward layers with .3 dropout for regularization
        self.EncoderFeedforward = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.ReLU(),
                                                nn.Linear(4 * n_embd, n_embd))

    # gets self attention for self. takes in optional entity info of dim (batch_size, entity_length, embedding_dim)
    def forward(self, x, entity_embeddings=None):
        if entity_embeddings is not None:
            x_with_entity = torch.cat((x, entity_embeddings), dim=1)
            len_entity = entity_embeddings.shape[1]
        else:
            x_with_entity = x  # use normal x if no entity info is added
            len_entity = 0

        attn_output, _ = self.EncoderAttention(x_with_entity, x_with_entity, x_with_entity)

        if entity_embeddings is not None:  # REMOVE extra entity info once its been used in attention
            attn_output = attn_output[:, :-len_entity,:]  # this will return our vector to (batch size, seq len, embedding dim)

        norm = self.Enorm1(x + attn_output) # resid connection and
        feedforward_output = self.EncoderFeedforward(norm)
        x = x + feedforward_output

        return x

# decoder self attention block
class DecoderLayers(nn.Module):
    """
    Args:
        n_embd: The embedding dimension
        n_head: The number of attention heads
        x: Decoder input sentence
        seq_len: Length of input x for making mask
        entity_info (Optional): Entity embeddings
        entity_len (Optional): Accounts for entity when creating mask

    Uses cross-attention with masking to create translation using Encoder's hidden states, entity embeddings, and decoder input
    """
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.DecoderAttention = nn.MultiheadAttention(n_embd, n_head, batch_first=True, dropout=0.1)

        # for self attention in decoder
        self.Dnorm1 = nn.LayerNorm(n_embd)
        self.Dnorm2 = nn.LayerNorm(n_embd)

        # feedforward layer with .3 dropout for regularization
        self.DecoderFeedforward = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.ReLU(),
                                                nn.Linear(4 * n_embd, n_embd))

    # gets self attention for decoder. takes in optional entity info of dim (batch_size, entity_length, embedding_dim)
    # creates mask inside function
    def forward(self, x, pad_mask, entity_embeddings=None):
        if entity_embeddings is not None:
            # Decoder entity info would need to be added to the beginning or else it would be masked out
            x_with_entity = torch.cat((entity_embeddings, x), dim=1) 
            len_entity = entity_embeddings.shape[1]
        else:
            x_with_entity = x
            len_entity = 0
        
        # either returns a normal self attention mask, or one modified for entity information
        # either returns seq_len, seq_len mask or (seq_len + entity_len), (seq_len + entity_len) with the diagonal shifter
        # this appends 0s (which nn.attention interprets as areas NOT to mask) to the beginning of the mask
        # if phrase is "who directed up", mask without entity info =
        # [0, 1, 1
        #  0, 0, 1
        #  0, 0, 0]
        # and new phrase is "who directed up <Oben>", mask would be
        # [0, 0, 1, 1
        #  0, 0, 0, 1
        #  0, 0, 0, 0] NOTE: in nn.Module, the 1s are masked out, not the 0s.

        # Attention mask
        mask = torch.triu(torch.ones(x.shape[1] + len_entity, x.shape[1] + len_entity, device=device), diagonal= len_entity +1).bool()

        attn_output, _ = self.DecoderAttention(x_with_entity, x_with_entity, x_with_entity, attn_mask=mask)

        if entity_embeddings is not None:  # REMOVE extra entity info once its been used in attention
            attn_output = attn_output[:, :-len_entity,
                          :]  # this will return our vector to (batch size, seq len, embedding dim)

        norm = self.Dnorm1(x + attn_output)  # resid connection and layer norm
        feedforward_output = self.DecoderFeedforward(norm)
        x = x + feedforward_output

        return x


# cross attention block
class CrossAttentionBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()

        self.CrossAttention = nn.MultiheadAttention(n_embd, n_head, batch_first=True, dropout=0.1)

        # for cross attention
        self.Cnorm1 = nn.LayerNorm(n_embd)
        self.Cnorm2 = nn.LayerNorm(n_embd)

        # feedforward layer with .3 dropout for regularization
        self.CrossFeedforward = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.ReLU(),
                                              nn.Linear(4 * n_embd, n_embd))
    
    def forward(self, decoder_input, encoder_output, pad_mask=None, entity_embeddings=None):
        # First concatenate entity embeddings with encoder output
        if entity_embeddings is not None:
            encoder_output_with_entity = torch.cat((entity_embeddings, encoder_output), dim=1)
        else:
            encoder_output_with_entity = encoder_output

        # Create or adjust padding mask to match encoder_output_with_entity length
        if pad_mask is not None:
            pad_mask = pad_mask.bool()  # i like da boolean masks bro
            if entity_embeddings is not None:
              
                # Create new padding mask matching the concatenated length
                entity_mask = torch.ones((pad_mask.shape[0], entity_embeddings.shape[1]), 
                                       dtype=torch.bool, device=pad_mask.device)
                pad_mask = torch.cat((entity_mask, pad_mask), dim=1)
            
            # debugging assertion: erify mask shape matches the sequence length
            assert pad_mask.shape[1] == encoder_output_with_entity.shape[1], \
                f"Padding mask length {pad_mask.shape[1]} doesn't match sequence length {encoder_output_with_entity.shape[1]}"

        # get cross-attention with key padding mask
        attn_output, _ = self.CrossAttention(
            decoder_input, 
            encoder_output_with_entity,
            encoder_output_with_entity,
            key_padding_mask=None if pad_mask is None else ~pad_mask.bool()
        )

        # apply residual connection and normalization
        x = self.Cnorm1(decoder_input + attn_output)
        feedforward_output = self.CrossFeedforward(x)
        x = self.Cnorm2(x + feedforward_output)

        return x



