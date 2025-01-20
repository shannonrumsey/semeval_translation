import torch
from torch import nn
from attention import EncoderLayers, DecoderLayers, CrossAttentionBlock
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


class PositionalEmbedding(nn.Module):
    """
    Takes in an embedded input and outputs the sum of the positional representations and the learned semantic embeddings

    In the case of entity embeddings, it takes in embedded entity tokens and returns the sum of the semantic embeddings
    and positional embeddings. The positional embeddings will be based only on the entity. So "Be yon ce" will have 3 positions,
    even if the actual occurance of this word occurs 12 indexes into a larger sentence

    entity positional and semabtic embeddings will be seperate objects from those of normal words, but they will use the same class

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

    def __init__(self, vocab_size, n_embd, n_head, n_layer, max_seq_len, max_entity_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.entity_embedding = nn.Embedding(vocab_size, n_embd)

        self.pos_embedding = PositionalEmbedding(n_embd, max_seq_len)
        self.entity_pos_embedding = PositionalEmbedding(n_embd, max_entity_len)

        # Note: here I code embeddings and entity embeddings separately, although you could also have them use the
        # same embeddings
        # pros of separating them: allows the model to optimize the embeddings for specific entity
        # info. For example, maybe attention will work better if the model optimizes these parameters separately in
        # such a way that it "tags" them as entities in their embeddings rather than just embedding them as normal words
        #
        # cons of separating them: more parameters, slower convergence, and potential fragmentation of representations,
        # where entity embeddings and word embeddings might not integrate well
        #
        # based on these pros and cons, I still think separating them is the better approach, but we can certainly
        # experiment with both if needed

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
        if entity_info is not None:
            pad_mask = (torch.cat((entity_info, x), dim=1) == semeval_train_dataset.vocab["<PAD>"])
        else:
            pad_mask = (x == semeval_train_dataset.vocab["<PAD>"])

        print(pad_mask)
        encoder_inputs = x
        x = self.embedding(x)
        x = self.pos_embedding(x)  # pos_embedding takes in the semantic embedding and manually sums them
        # ^ this comment is for Darian because I keep forgetting this and rereading the code XD

        entity_embeddings = self.entity_embedding(entity_info) if entity_info is not None else None
        entity_embeddings = self.entity_pos_embedding(entity_embeddings) if entity_embeddings is not None else None

        # Mimics PyTorch's TransformerEncoder
        for attention_layer in self.attention_layers:
            # if entity info is provided, it will do cross attention using x + entity info as the keys and values and x as the query
            # if no info is provided, it will just do normal self attention
            x = attention_layer(x, x, pad_mask=pad_mask, entity_embeddings=entity_embeddings)

        return x, entity_embeddings, encoder_inputs


class TransformerDecoder(nn.Module):
    """
    Requires masking for self-attention
    """

    def __init__(self, max_seq_length, max_entity_length, n_embd, n_head, vocab_size, attention_layers=5):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)  # will give us token embeddings
        self.position_embedding_table = nn.Embedding(max_seq_length,
                                                     n_embd)
        self.last_linear_layer = nn.Linear(n_embd, vocab_size)

        self.self_attention_layers = nn.ModuleList(
            [DecoderLayers(n_embd, n_head) for _ in range(attention_layers)]
        )
        self.cross_attention_layers = nn.ModuleList(
            [CrossAttentionBlock(n_embd, n_head) for _ in range(attention_layers)]
        )

        # I am not sure if we want to use separate entity embeddings in the decoder or reuse the ones from the encoder.
        # I think this is something that we need to experiment with to know for sure, thus I will include
        # both options in this code

        # separate entity embeddings for just the decoder
        self.entity_token_embedding_table = nn.Embedding(vocab_size, n_embd)  # will give us token embeddings
        self.entity_position_embedding_table = nn.Embedding(max_entity_length,
                                                            n_embd)

    def forward(self, decoder_input, encoder_output, encoder_inputs, encoder_entity_embeddings=None, entity_info=None,
                use_encoders_entities=False):
        """
        Uses cross-attention and self-attention
        Args:
            decoder_input (batch size, seq_len): Input to Decoder
            encoder_output (batch_size, seq_len, embedding dim): hidden states of encoder. Used in cross attention with decoder
            encoder_entity_embeddings (batch_size, entity_seq_len, embedding dim): The trained entity embeddings from the encoder
            entity_info (entity_seq_length): The unprocessed encoded entity tokens. These will be used to make a separate entity representation in the decoder
            use_encoders_entities (bool): Describes whether we want to reuse the entity embeddings from the encoder or create new ones
        Returns:
            Translated Sentence (batch_size, seq_len, vocab_size)
        Notes:
            this code allows for the option to either use the encoders entity embeddings or generate those specific to the decoder by setting
            use_encoders_entities to true or false respectively.

            If use_encoders_entities = true, the model will use the same entity embeddings as used in the encoder and will
            ignore entity_info (which is the raw, unembedded entities)

            If use_encoders_entities = false, the model will ignore the encoder_entity_embeddings and
            will construct its own embeddings using entity_info (which is the raw, unembedded entities)

            Thus, this parameter allows us to easily switch between approaches for experimentation.

            I think we should experiment with both approaches to see which yeilds
            better results
        """
        seq_len = decoder_input.size(1)
        if entity_info is not None:
            entity_len = entity_info.size(1)

        # embedd the decoder input 
        token_embeddings = self.token_embedding_table(decoder_input)  # batch, seq len, embedding size

        position_embeddings = self.position_embedding_table(torch.arange(seq_len, device=device))

        x = token_embeddings + position_embeddings  # adding token and position embeddings

        if entity_info != None and use_encoders_entities is False:
            # if use_encoders_entities is False, then we create new embeddings in the decoder

            entity_token_embeddings = self.entity_token_embedding_table(decoder_input)  # batch, entity seq len, embedding size

            entity_position_embeddings = self.entity_position_embedding_table(torch.arange(entity_len, device=device))

            entity_embeddings = entity_token_embeddings + entity_position_embeddings  # adding token and position embeddings

        elif encoder_entity_embeddings != None and use_encoders_entities is True:
            # if use_encoders_entities is True, then we use the same entities as the encoder
            entity_embeddings = encoder_entity_embeddings
            
        else:
            entity_embeddings = None 

        # pass through self-attention layers
        for self_attn_block in self.self_attention_layers:
            if entity_info is not None:
                pad_mask = (torch.cat((entity_info, decoder_input), dim=1) == semeval_train_dataset.vocab["<PAD>"])
            else:
                pad_mask = (decoder_input == semeval_train_dataset.vocab["<PAD>"])
            x = self_attn_block(x, pad_mask, entity_embeddings)

        # pass through cross-attention layers
        for cross_attn_block in self.cross_attention_layers:
            if entity_info is not None:
                pad_mask = (torch.cat((entity_info, encoder_inputs), dim=1) == semeval_train_dataset.vocab["<PAD>"])
            else:
                pad_mask = (encoder_inputs == semeval_train_dataset.vocab["<PAD>"])
            x = cross_attn_block(x, encoder_output, pad_mask, entity_embeddings)

        # final linear layer to map to vocab size
        output = self.last_linear_layer(x)  # (batch_size, seq_len, vocab_size)

        return output
