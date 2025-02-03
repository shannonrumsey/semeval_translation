import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3,4'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import setproctitle
setproctitle.setproctitle('semeval_exp')

from dataset import pretrain_dataset, semeval_train_dataset, semeval_val_dataset, semeval_train_loader, semeval_val_loader, pretrain_train_loader, pretrain_val_loader
import torch
from torch import nn
from torch import optim
from model import PositionalEmbedding, TransformerEncoder, TransformerDecoder, device
import sys
from config import MODEL_CONFIG
from util_funcs import find_max_sequence_length

#### Attention: This experiment is where the encoder creates embeddings and 
# decoder creates own embeddings for self attention and cross attention


"""
entity_info should be batches of entities, corresponding to the input data
"""
vocab_size = len(pretrain_dataset.vocab)

pretrain_encoder_path = os.path.join(os.path.dirname(__file__), "trained_models/pretrain_encoder_model")
pretrain_decoder_path = os.path.join(os.path.dirname(__file__), "trained_models/pretrain_decoder_model")
train_encoder_path = os.path.join(os.path.dirname(__file__), "trained_models/train_encoder_model_exp2")
train_decoder_path = os.path.join(os.path.dirname(__file__), "trained_models/train_decoder_model_exp2")

max_seq_len_pretrain = find_max_sequence_length(dataset=pretrain_dataset)
max_seq_len_train = find_max_sequence_length(dataset=semeval_train_dataset)
max_seq_len_val = find_max_sequence_length(dataset=semeval_val_dataset)
max_seq_len = max(max_seq_len_pretrain, max_seq_len_train, max_seq_len_val)
entity_len_train = find_max_sequence_length(dataset=semeval_train_dataset, entity = True)
entity_len_val = find_max_sequence_length(dataset=semeval_val_dataset, entity = True)

entity_len = max(entity_len_train, entity_len_val)

# Check what values were used during pretraining
print("Loading pretrained model with configuration:")
pretrained_state = torch.load(pretrain_encoder_path)
print(f"Pretrained model hidden size: {pretrained_state['attention_layers.0.CrossAttention.out_proj.weight'].size(0)}")

# Adjust n_embd to match pretrained model
n_embd = 128  # Update this to match pretrained model
n_head = MODEL_CONFIG['n_head']
n_layer = MODEL_CONFIG['n_layer']


def run_model(n_embd, n_head, n_layer, train_loader, val_loader, pretrain_encoder_path, pretrain_decoder_path, train_encoder_path=None, train_decoder_path=None, train=False):
   
   
    def decode_tokens(token_ids):
        """Helper function to convert token ids back to text"""
        vocab = pretrain_dataset.vocab
        pad_token = vocab["<PAD>"]
        # Create reverse vocabulary (id to token)
        id_to_token = {v: k for k, v in vocab.items()}
        # Convert tensor to list and filter out padding tokens
        token_list = [int(id) for id in token_ids if int(id) != pad_token]
        # Convert tokens to words using reverse vocabulary
        words = [id_to_token[id] for id in token_list]
        return ' '.join(words)
        
    pos = PositionalEmbedding(n_embd)
    encoder = TransformerEncoder(vocab_size=vocab_size,
                                n_embd=n_embd,
                                n_head=n_head,
                                n_layer=n_layer,
                                max_seq_len=max_seq_len,
                                max_entity_len=entity_len).to(device)
    decoder = TransformerDecoder(max_seq_length=max_seq_len,
                                n_embd=n_embd,
                                n_head=n_head,
                                vocab_size=vocab_size,
                                max_entity_len=entity_len).to(device)
    pad_index = pretrain_dataset.vocab["<PAD>"]
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_index)
    enc_optimizer = optim.AdamW(encoder.parameters(), lr=0.001)
    dec_optimizer = optim.AdamW(decoder.parameters(), lr=0.001)
    
    if train:
        encoder_path = train_encoder_path
        decoder_path = train_decoder_path


        encoder.load_state_dict(torch.load(pretrain_encoder_path, weights_only=True))
        decoder.load_state_dict(torch.load(pretrain_decoder_path, weights_only=True))


    else: 
        encoder_path = pretrain_encoder_path
        decoder_path = pretrain_decoder_path
    
    num_epoch = 50
    prev_loss = None
    for epoch in range(num_epoch):
        epoch_loss = 0
        encoder.train()
        decoder.train()

        for batch in train_loader:
            if len(batch) == 4: # we don't have any entity info
                encoder_input, decoder_input, target, mask = batch
                entity_info = None
            elif len(batch) == 5: # we have entity info
                encoder_input, decoder_input, target, mask, entity_info = batch

            # Convert inputs to long (integer) type
            encoder_input = encoder_input.long().to(device)
            decoder_input = decoder_input.long().to(device)
            target = target.long().to(device)
            mask = mask.to(device)
            if entity_info is not None:
                entity_info = entity_info.long().to(device)

            enc_optimizer.zero_grad()
            dec_optimizer.zero_grad()

            hidden_states, encoder_entity_embeddings, encoder_inputs = encoder(encoder_input, 
                                                                 entity_info=entity_info,
                                                                 padding_mask=mask)
            
            # does not use encoder entity embeddings, but instead creates its own
            # entity embeddings for self attention
            decoder_outputs = decoder(decoder_input, 
                                    hidden_states, 
                                    encoder_inputs,
                                    encoder_entity_embeddings=None,
                                    entity_info=entity_info,
                                    use_encoders_entities=False,
                                    entities_in_self_attn=True,
                                    padding_mask=mask)

            loss = loss_fn(decoder_outputs.view(-1, vocab_size), target.view(-1))

            loss.backward()
            epoch_loss += loss.item()

            enc_optimizer.step()
            dec_optimizer.step()

        avg_pretrain_loss = epoch_loss / len(train_loader)
        print(f"Epoch Training Loss: {avg_pretrain_loss}")

        val_loss = 0
        encoder.eval()
        decoder.eval()
        print(f"\nEpoch {epoch} Translation Samples:")
        with torch.no_grad():
            # Print first 3 samples from validation set
            sample_count = 0
            for batch in val_loader:
                if sample_count >= 3:  # Only show 3 samples
                    break
                    
                if len(batch) == 4:
                    encoder_input, decoder_input, target, mask = batch
                    entity_info = None
                elif len(batch) == 5:
                    encoder_input, decoder_input, target, mask, entity_info = batch

                # Convert inputs to long (integer) type
                encoder_input = encoder_input.long().to(device)
                decoder_input = decoder_input.long().to(device)
                target = target.long().to(device)
                mask = mask.to(device)
                if entity_info is not None:
                    entity_info = entity_info.long().to(device)

                # Get model outputs
                hidden_states, encoder_entity_embeddings, encoder_inputs = encoder(
                    encoder_input, entity_info=entity_info, padding_mask=mask)
                decoder_outputs = decoder(
                    decoder_input, hidden_states, encoder_inputs,
                    encoder_entity_embeddings=None, entity_info=entity_info,
                    use_encoders_entities=False, entities_in_self_attn=True,
                    padding_mask=mask)

                # Get predictions
                predictions = torch.argmax(decoder_outputs, dim=-1)

                # Print samples
                for i in range(min(2, encoder_input.size(0))):  # Print up to 2 samples from this batch
                    print("\nSample", sample_count + 1)
                    print("Input:", decode_tokens(encoder_input[i].cpu()))
                    if entity_info is not None:
                        print("Entity:", decode_tokens(entity_info[i].cpu()))
                    print("Target:", decode_tokens(target[i].cpu()))
                    print("Predicted:", decode_tokens(predictions[i].cpu()))
                    print("-" * 50)
                    sample_count += 1
                    if sample_count >= 3:
                        break

                # Continue with validation loss calculation
                loss = loss_fn(decoder_outputs.view(-1, vocab_size), target.view(-1))
                val_loss += loss.item()

            total_val_loss = val_loss / len(val_loader)
            print(f"\nValidation Loss on Epoch {epoch}: {total_val_loss}")

            # Save model with lowest loss
            if prev_loss is None or total_val_loss < prev_loss:
                print("LOWEST LOSS: SAVING MODEL")
                torch.save(encoder.state_dict(), encoder_path)
                torch.save(decoder.state_dict(), decoder_path)
                prev_loss = total_val_loss

# Darian's extend function
def extend_embedding(embedding, target_size, scale=0.01):
    current_size, dim = embedding.shape
    if current_size < target_size:
        additional_weights = torch.randn(target_size - current_size, dim, device=device, dtype=torch.float32) * scale
        embedding = torch.cat([embedding, additional_weights], dim=0)
    return embedding

encoder = torch.load("trained_models/pretrain_encoder_model", map_location=device)
decoder = torch.load("trained_models/pretrain_decoder_model", map_location=device)
print("printing original dtype")
print(encoder["pos_embedding.pos_embedding.weight"].dtype)
encoder["pos_embedding.pos_embedding.weight"] = extend_embedding(encoder["pos_embedding.pos_embedding.weight"], max_seq_len)
encoder["entity_pos_embedding.pos_embedding.weight"] = extend_embedding(encoder["entity_pos_embedding.pos_embedding.weight"], entity_len)
decoder["pos_embedding.pos_embedding.weight"] = extend_embedding(decoder["pos_embedding.pos_embedding.weight"], max_seq_len)
decoder["entity_pos_embedding.pos_embedding.weight"] = extend_embedding(decoder["entity_pos_embedding.pos_embedding.weight"], entity_len)
torch.save(encoder, "trained_models/pretrain_encoder_model_extended")
torch.save(decoder, "trained_models/pretrain_decoder_model_extended")
pretrain_encoder_path = "trained_models/pretrain_encoder_model_extended"
pretrain_decoder_path = "trained_models/pretrain_decoder_model_extended"

# Pretrain model
# run_model(n_embd, n_head, n_layer, train_loader=pretrain_train_loader,
#           val_loader=pretrain_val_loader, pretrain_encoder_path=pretrain_encoder_path,
#           pretrain_decoder_path=pretrain_decoder_path, train=False)

# Train model
run_model(n_embd, n_head, n_layer, train_loader=semeval_train_loader,
          val_loader=semeval_val_loader, pretrain_encoder_path=pretrain_encoder_path,
          pretrain_decoder_path=pretrain_decoder_path, train_encoder_path=train_encoder_path,
          train_decoder_path=train_decoder_path, train=True)
