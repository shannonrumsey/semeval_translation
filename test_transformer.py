import torch
import unittest
from model import TransformerEncoder, TransformerDecoder, device

class TestTransformer(unittest.TestCase):
    def setUp(self):
        # Test configuration
        self.batch_size = 2
        self.seq_length = 5
        self.vocab_size = 1000
        self.n_embd = 128
        self.n_head = 4
        self.n_layer = 2
        self.entity_len = 3

        # creates a padding mask where last two tokens are padding (1 for valid tokens, 0 for padding)
        self.padding_mask = torch.ones((self.batch_size, self.seq_length)).to(device)
        self.padding_mask[:, -2:] = 0  # mark last two positions as padding
        
        # create inputs with actual padding tokens
        self.pad_token = 0
        self.encoder_input = torch.randint(1, self.vocab_size, (self.batch_size, self.seq_length)).to(device)
        self.encoder_input[:, -2:] = self.pad_token  # add padding tokens as last two tokens
        
        self.decoder_input = torch.randint(1, self.vocab_size, (self.batch_size, self.seq_length)).to(device)
        self.decoder_input[:, -2:] = self.pad_token

        # use sample entity info
        self.entity_info = torch.randint(0, self.vocab_size, (self.batch_size, self.entity_len)).to(device)

        # Initialize models
        self.encoder = TransformerEncoder(
            vocab_size=self.vocab_size,
            n_embd=self.n_embd,
            n_head=self.n_head,
            n_layer=self.n_layer,
            max_seq_len=self.seq_length,
            max_entity_len=self.entity_len
        ).to(device)

        self.decoder = TransformerDecoder(
            max_seq_length=self.seq_length,
            n_embd=self.n_embd,
            n_head=self.n_head,
            vocab_size=self.vocab_size,
            max_entity_len=self.entity_len
        ).to(device)

    def test_padding_mask_shape(self):
        """Test if the padding mask has correct shape and is being accepted"""
        try:
            hidden_states, encoder_entity_embeddings, encoder_inputs = self.encoder(
                self.encoder_input,
                entity_info=self.entity_info,
                padding_mask=self.padding_mask
            )
            self.assertTrue(True, "Encoder accepted padding mask")
        except Exception as e:
            self.fail(f"Encoder failed to process padding mask: {str(e)}")

    def test_padding_mask_effect(self):
        """Test if padding mask is actually affecting attention scores"""
        with torch.no_grad():
            # Create two identical inputs except for padding
            input1 = self.encoder_input.clone()
            input2 = self.encoder_input.clone()
            input2[:, -2:] = torch.randint(1, self.vocab_size, (self.batch_size, 2)).to(device)  # Different tokens in padding positions
            
            # Run both through encoder
            hidden_states1, _, _ = self.encoder(input1, entity_info=self.entity_info)
            hidden_states2, _, _ = self.encoder(input2, entity_info=self.entity_info)
            
            # The hidden states should be the same in non-padding positions
            non_padding_states1 = hidden_states1[:, :-2, :]
            non_padding_states2 = hidden_states2[:, :-2, :]
            self.assertTrue(torch.allclose(non_padding_states1, non_padding_states2, rtol=1e-4),
                           "Padding mask not working: non-padding positions differ")

    def test_end_to_end_with_mask(self):
        """Test full encoder-decoder pipeline with padding mask"""
        try:
            # Forward pass through encoder
            hidden_states, encoder_entity_embeddings, encoder_inputs = self.encoder(
                self.encoder_input,
                entity_info=self.entity_info,
                padding_mask=self.padding_mask
            )
            
            # Forward pass through decoder
            output = self.decoder(
                self.decoder_input,
                hidden_states,
                encoder_inputs,
                encoder_entity_embeddings=None,
                entity_info=self.entity_info,
                use_encoders_entities=False,
                entities_in_self_attn=True,
                padding_mask=self.padding_mask
            )
            
            # Check output shape
            expected_shape = (self.batch_size, self.seq_length, self.vocab_size)
            self.assertEqual(output.shape, expected_shape,
                           f"Expected output shape {expected_shape}, got {output.shape}")
            
        except Exception as e:
            self.fail(f"End-to-end test failed: {str(e)}")

if __name__ == '__main__':
    unittest.main() 