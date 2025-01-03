'''
This file contains code for the decoder self attention, encoder self attention, and cross attention

All attention functions allow the user to pass in additional entity info. This is how that info is incorporated:

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

'''

import torch
import torch.nn as nn

class EncoderAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        """
        :param embed_dim: the embedding dim
        :param num_heads: the number of attention heads

        Function preforms unmasked self attention, complete with a feedforward layer, layer normalization, and residual connections
        Function gives option to include entity information
        """
        super(TransformerBlock, self).__init__()
        self.EncoderAttention = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True, dropout=0.2)

        # for self attention in encoder
        self.Enorm1 = nn.LayerNorm(embed_dim)
        self.Enorm2 = nn.LayerNorm(embed_dim)

        # feedforward layers with .3 dropout for regularization
        self.EncoderFeedforward = nn.Sequential(nn.Linear(embed_dim, 4 * embed_dim), nn.ReLU(), nn.Dropout(0.3),
                                                nn.Linear(4 * embed_dim, embed_dim))


    # gets self attention for self. takes in optional entity info of dim (batch_size, entity_length, embedding_dim)
    def forward(self, x, entity_info = None):
        if entity_info is not None:
            x_with_entity = torch.cat((x, entity_info), dim=1)
            len_entity = entity_info.shape[1]
        else:
            x_with_entity = x  # use normal x if no entity info is added
            len_entity = 0
        attn_output, _ = self.EncoderAttentionattention(x_with_entity, x_with_entity, x_with_entity)

        if entity_info is not None:  # REMOVE extra entity info once its been used in attention
            attn_output = attn_output[:, :-len_entity,:]  # this will return our vector to (batch size, seq len, embedding dim)

        x = self.Enorm1(x + attn_output) # resid connection and
        feedforward_output = self.EncoderFeedforward(x)
        x = self.Enorm2(x + feedforward_output)
        return x

# decoder self attention block
class DecoderAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(TransformerBlock, self).__init__()

        self.DecoderAttention = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True, dropout=0.2)


        # for self attention in decoder
        self.Dnorm1 = nn.LayerNorm(embed_dim)
        self.Dnorm2 = nn.LayerNorm(embed_dim)



        # feedforward layer with .3 dropout for regularization

        self.DecoderFeedforward = nn.Sequential(nn.Linear(embed_dim, 4 * embed_dim), nn.ReLU(), nn.Dropout(0.3),
                                                nn.Linear(4 * embed_dim, embed_dim))





    def create_attention_mask(self, seq_len, entity_len):
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

        mask = torch.triu(torch.ones(seq_len + entity_len, seq_len + entity_len, device=device), diagonal= entity_len +1).bool()
        return mask

    # gets self attention for decoder. takes in optional entity info of dim (batch_size, entity_length, embedding_dim)
    # creates mask inside function
    def forward(self, x, entity_info = None):
        if entity_info is not None:
            x_with_entity = torch.cat((entity_info, x), dim=1) # in decoder entity info would need to be added to the beginning or else it would be masked out
            len_entity = entity_info.shape[1]
        else:
            x_with_entity = x
            len_entity = 0
        mask = self.create_attention_mask(seq_len=x.shape[1], entity_len=len_entity)
        attn_output, _ = self.DecoderAttention(x_with_entity, x_with_entity, x_with_entity, attn_mask = mask)


        if entity_info is not None:  # REMOVE extra entity info once its been used in attention
            attn_output = attn_output[:, :-len_entity,
                          :]  # this will return our vector to (batch size, seq len, embedding dim)

        x = self.Dnorm1(x + attn_output)  # resid connection and layer norm
        feedforward_output = self.DecoderFeedforward(x)
        x = self.Dnorm2(x + feedforward_output)
        return x





# cross attention block
class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(TransformerBlock, self).__init__()

        self.CrossAttention = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True, dropout=0.2)


        # for cross attention
        self.Cnorm1 = nn.LayerNorm(embed_dim)
        self.Cnorm2 = nn.LayerNorm(embed_dim)

        # feedforward layer with .3 dropout for regularization

        self.CrossFeedforward = nn.Sequential(nn.Linear(embed_dim, 4 * embed_dim), nn.ReLU(), nn.Dropout(0.3),
                                              nn.Linear(4 * embed_dim, embed_dim))


    def forward(self, decoder_input, encoder_output, entity_info=None):
            # concatenate entity_info to the encoder inputs if provided
            if entity_info is not None:
                encoder_output_with_entity = torch.cat((entity_info, encoder_output), dim=1)
                len_entity = entity_info.shape[1]
            else:
                encoder_output_with_entity = encoder_output
                len_entity = 0


            mask = self.create_attention_mask(seq_len=decoder_input.shape[1], entity_len=len_entity)

            # get cross-attention (decoder query, encoder key & value)
            attn_output, _ = self.CrossAttention(decoder_input_with_entity, encoder_output_with_entity,
                                                 encoder_output_with_entity, attn_mask=mask)
            # output will be of size: (batch_size, seq_len_decoder, embed_dim)
            # no need to remove the entity info because it was in the encoder. (only used as a key and not a query)


            # apply residual connection and normalization
            x = self.CrossNorm1(decoder_input + attn_output)
            feedforward_output = self.CrossFeedforward(x)
            x = self.CrossNorm2(x + feedforward_output)

            return x


