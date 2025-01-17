from dataset import pretrain_dataset, semeval_dataset, pretrain_train_loader, pretrain_val_loader, semeval_train_loader, semeval_val_loader, entity_train_info, entity_val_info
import os
import torch
from torch import nn
from torch import optim
from model import PositionalEmbedding, TransformerEncoder, TransformerDecoder, device
import sys



"""
entity_info should be batches of entities, corresponding to the input data
"""
vocab_size = len(pretrain_dataset.vocab)
# in order to set the proper size for max seq length for our positional embeddings
def find_max_sequence_length(dataset=None, entity_info=None):
    if entity_info:
        longest_entity = max(len(row) for batch in entity_info for row in batch)
        return longest_entity
    else:
        for ids in dataset.corpus_encoder_ids:
            print(len(ids))
        encoder_max = max(len(ids) for ids in dataset.corpus_encoder_ids)
        decoder_max = max(len(ids) for ids in dataset.corpus_decoder_ids)
        target_max = max(len(ids) for ids in dataset.corpus_target_ids)
        return max(encoder_max, decoder_max, target_max)

max_seq_len_pretrain = find_max_sequence_length(dataset=pretrain_dataset)
max_seq_len_train = find_max_sequence_length(dataset=semeval_dataset)
max_seq_len = max(max_seq_len_pretrain, max_seq_len_train)
entity_len = find_max_sequence_length(entity_info=entity_info)

n_embd = 128
n_head = 4
n_layer = 6

def run_model(n_embd, n_head, n_layer, train_loader, val_loader, pretrain_encoder_path, pretrain_decoder_path, train_encoder_path=None, train_decoder_path=None, train=False):
    pos = PositionalEmbedding(n_embd)
    encoder = TransformerEncoder(vocab_size=vocab_size,
                                n_embd=n_embd,
                                n_head=n_head,
                                n_layer=n_layer,
                                max_seq_len=max_seq_len,
                                entity_len=entity_len).to(device)
    decoder = TransformerDecoder(max_seq_length=max_seq_len,
                                n_embd=n_embd,
                                n_head=n_head,
                                vocab_size=vocab_size,
                                entity_len=entity_len).to(device)
    pad_index = pretrain_dataset.vocab["<PAD>"]
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_index)
    enc_optimizer = optim.AdamW(encoder.parameters(), lr=0.001)
    dec_optimizer = optim.AdamW(encoder.parameters(), lr=0.001)
    
    if train:
        encoder_path = train_encoder_path
        decoder_path = train_decoder_path
        encoder.load_state_dict(torch.load(pretrain_encoder_path, weights_only=True))
        decoder.load_state_dict(torch.load(pretrain_decoder_path, weights_only=True))
    else: 
        encoder_path = pretrain_encoder_path
        decoder_path = pretrain_decoder_path
    
    num_epoch = 1
    prev_loss = None
    for epoch in range(num_epoch):
        epoch_loss = 0
        encoder.train()
        decoder.train()
        for encoder_input, decoder_input, target, mask in train_loader:
            encoder_input = encoder_input.to(device)
            decoder_input = decoder_input.to(device)
            target = target.to(device)
            mask = mask.to(device)

            enc_optimizer.zero_grad()
            dec_optimizer.zero_grad()

            hidden_states = encoder(encoder_input)
            decoder_outputs = decoder(decoder_input, hidden_states)

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
        with torch.no_grad():
            for encoder_input, decoder_input, target, mask in val_loader:
                encoder_input = encoder_input.to(device)
                decoder_input = decoder_input.to(device)
                target = target.to(device)
                mask = mask.to(device)

                hidden_states = encoder(encoder_input)
                decoder_outputs = decoder(decoder_input, hidden_states)

                loss = loss_fn(decoder_outputs.view(-1, vocab_size), target.view(-1))
                val_loss += loss.item()

            total_val_loss = val_loss/ len(val_loader)
            print(f"Pretrain Validation Loss on Epoch {epoch}: {total_val_loss}")

            # Save model with lowest loss
            if prev_loss is None or total_val_loss < prev_loss:
                print("LOWEST LOSS: SAVING MODEL")
                torch.save(encoder.state_dict(), encoder_path)
                torch.save(decoder.state_dict(), decoder_path)
                prev_loss = total_val_loss

        
pretrain_encoder_path = os.path.join(os.path.dirname(__file__), "trained_models/pretrain_encoder_model")
pretrain_decoder_path = os.path.join(os.path.dirname(__file__), "trained_models/pretrain_decoder_model")
train_encoder_path = os.path.join(os.path.dirname(__file__), "trained_models/train_encoder_model")
train_decoder_path = os.path.join(os.path.dirname(__file__), "trained_models/train_decoder_model")

# Pretrain model
run_model(n_embd, n_head, n_layer, train_loader=pretrain_train_loader,
          val_loader=pretrain_val_loader, pretrain_encoder_path=pretrain_encoder_path,
          pretrain_decoder_path=pretrain_decoder_path, train=False)

# Train model
run_model(n_embd, n_head, n_layer, train_loader=semeval_train_loader,
          val_loader=semeval_val_loader, pretrain_encoder_path=pretrain_encoder_path,
          pretrain_decoder_path=pretrain_decoder_path, train_encoder_path=train_encoder_path,
          train_decoder_path=train_decoder_path, train=True)
