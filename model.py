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
        self.pos_embedding = PositionalEmbedding(n_embd, max_seq_len)

        self.entity_embedding = nn.Embedding(vocab_size, n_embd)
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

    def forward(self, x, entity_info=None, padding_mask=None):
        """
        Args:
            x (batch size, seq_len): Input sentences
            entity_info (batch size, entity len): Encoded entity embeddings
            padding_mask (batch size, seq_len): Binary mask where 0 indicates padding
        Returns:
            -hidden states from encoder, entity embeddings, and encoder inputs
        """
        encoder_inputs = x
        x = self.embedding(x)
        x = self.pos_embedding(x)

        # when using entity embeddings ...
        if entity_info is not None:
            entity_embeddings = self.entity_embedding(entity_info)
            entity_embeddings = self.entity_pos_embedding(entity_embeddings)

            # Concatenate entity embeddings with input embeddings before attention
            x = torch.cat((entity_embeddings, x), dim=1)
            
            # Create or adjust padding mask to match x's sequence length
            total_length = x.shape[1]  # Get the actual sequence length after concatenation
            
            if padding_mask is None:
                # Create new padding mask for concatenated sequence
                padding_mask = torch.zeros((x.shape[0], total_length), dtype=torch.bool, device=x.device)
                # Set padding for input portion
                input_start = entity_embeddings.shape[1]
                padding_mask[:, input_start:] = (encoder_inputs == semeval_train_dataset.vocab["<PAD>"]).bool()
            else:
                # Create new padding mask of correct size
                new_padding_mask = torch.zeros((x.shape[0], total_length), dtype=torch.bool, device=padding_mask.device)
                # Copy original padding mask to input portion
                input_start = entity_embeddings.shape[1]
                min_length = min(padding_mask.shape[1], x.shape[1] - input_start)
                new_padding_mask[:, input_start:input_start + min_length] = padding_mask[:, :min_length]
                # Fill remaining positions with True (masked)
                if input_start + min_length < x.shape[1]:
                    new_padding_mask[:, input_start + min_length:] = True
                padding_mask = new_padding_mask

        else:
            entity_embeddings = None
            if padding_mask is None:
                padding_mask = (encoder_inputs == semeval_train_dataset.vocab["<PAD>"]).bool()
            else:
                padding_mask = padding_mask.bool()

        # Pass through attention layers
        for attention_layer in self.attention_layers:
            x = attention_layer(x, x, pad_mask=padding_mask, entity_embeddings=None)

        # If we concatenated entity embeddings, split them back out
        if entity_info is not None:
            x = x[:, entity_embeddings.shape[1]:]  # Remove entity portion
        
        return x, entity_embeddings, encoder_inputs

    

class TransformerDecoder(nn.Module):
    """
    Requires masking for self-attention
    """

    def __init__(self, max_seq_length, n_embd, n_head, vocab_size, max_entity_len, attention_layers=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)  # will give us token embeddings
        self.pos_embedding = PositionalEmbedding(n_embd,
                                                     max_seq_length)
        self.last_linear_layer = nn.Linear(n_embd, vocab_size)

        self.self_attention_layers = nn.ModuleList(
            [DecoderLayers(n_embd, n_head) for _ in range(attention_layers)]
        )
        self.cross_attention_layers = nn.ModuleList(
            [CrossAttentionBlock(n_embd, n_head) for _ in range(attention_layers)]
        )

        # separate entity embeddings for just the decoder
        self.entity_embedding = nn.Embedding(vocab_size, n_embd)  # will give us token embeddings
        self.entity_pos_embedding = PositionalEmbedding(n_embd, max_entity_len)

    def forward(self, decoder_input, encoder_output, encoder_inputs, encoder_entity_embeddings=None, 
                entity_info=None, use_encoders_entities=False, entities_in_self_attn=True, padding_mask=None):
        """
        Args:
            decoder_input (batch size, seq_len): Input to Decoder
            encoder_output (batch_size, seq_len, embedding dim): hidden states of encoder. Used in cross attention with decoder
            encoder_entity_embeddings (batch_size, entity_seq_len, embedding dim): The trained entity embeddings from the encoder
            entity_info (entity_seq_length): The unprocessed encoded entity tokens. These will be used to make a separate entity representation in the decoder
            use_encoders_entities (bool): Describes whether we want to reuse the entity embeddings from the encoder or create new ones
            padding_mask (batch size, seq_len): Binary mask where 0 indicates padding
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
        # If no padding mask provided, create from PAD tokens
        if padding_mask is None:
            if entity_info is not None:
                padding_mask = (torch.cat((entity_info, decoder_input), dim=1) == semeval_train_dataset.vocab["<PAD>"]).bool()
            else:
                padding_mask = (decoder_input == semeval_train_dataset.vocab["<PAD>"]).bool()
        else:
            padding_mask = padding_mask.bool()  # d double dog checking that shi is boolean

        # embed the decoder input
        x = self.embedding(decoder_input)
        x = self.pos_embedding(x)

        if entity_info is not None and not use_encoders_entities:
            entity_embeddings = self.entity_embedding(entity_info)
            entity_embeddings = self.entity_pos_embedding(entity_embeddings)
        elif encoder_entity_embeddings is not None and use_encoders_entities:
            entity_embeddings = encoder_entity_embeddings
        else:
            entity_embeddings = None

        # Self-attention layers
        if entities_in_self_attn:
            for self_attn_block in self.self_attention_layers:
                x = self_attn_block(x, padding_mask, entity_embeddings)
        else:
            for self_attn_block in self.self_attention_layers:
                x = self_attn_block(x, padding_mask, entity_embeddings=None)

        # Cross-attention layers
        encoder_padding_mask = (encoder_inputs == semeval_train_dataset.vocab["<PAD>"]).bool()
        for cross_attn_block in self.cross_attention_layers:
            x = cross_attn_block(x, encoder_output, encoder_padding_mask, entity_embeddings)

        output = self.last_linear_layer(x)
        return output
