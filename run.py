from dataset import pretrain_dataset, semeval_train_dataset, semeval_val_dataset, semeval_train_loader, semeval_val_loader, pretrain_train_loader, pretrain_val_loader
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
def find_max_sequence_length(dataset, entity= False): # if entity == True, it will return the max entitry length
    if entity:
        if dataset.entity_ids != None:

            longest_entity = max(len(ids) for ids in dataset.entity_ids)
        return longest_entity
    else:
        for ids in dataset.corpus_encoder_ids:
            print(len(ids))
        encoder_max = max(len(ids) for ids in dataset.corpus_encoder_ids)
        decoder_max = max(len(ids) for ids in dataset.corpus_decoder_ids)
        target_max = max(len(ids) for ids in dataset.corpus_target_ids)
        return max(encoder_max, decoder_max, target_max)

max_seq_len_pretrain = find_max_sequence_length(dataset=pretrain_dataset)
max_seq_len_train = find_max_sequence_length(dataset=semeval_train_dataset)
max_seq_len_val = find_max_sequence_length(dataset=semeval_val_dataset)
max_seq_len = max(max_seq_len_pretrain, max_seq_len_train, max_seq_len_val)
entity_len_train = find_max_sequence_length(dataset=semeval_train_dataset, entity = True)
entity_len_val = find_max_sequence_length(dataset=semeval_val_dataset, entity = True)

entity_len = max(entity_len_train, entity_len_val)

n_embd = 512
n_head = 4
n_layer = 6

def run_model(n_embd, n_head, n_layer, train_loader, val_loader, pretrain_encoder_path, pretrain_decoder_path, train_encoder_path=None, train_decoder_path=None, train=False):
    pos = PositionalEmbedding(n_embd)
    encoder = TransformerEncoder(vocab_size=vocab_size,
                                n_embd=n_embd,
                                n_head=n_head,
                                n_layer=n_layer,
                                max_seq_len=max_seq_len,
                                max_entity_len=entity_len).to(device)
    decoder = TransformerDecoder(max_seq_length=max_seq_len,
                                max_entity_length= entity_len,
                                n_embd=n_embd,
                                n_head=n_head,
                                vocab_size=vocab_size).to(device)
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
    
    num_epoch = 50
    prev_loss = None
    for epoch in range(num_epoch):
        epoch_loss = 0
        encoder.train()
        decoder.train()

        for batch in train_loader:

            # for our mental health and sanity 
            # print(
            #     "\n\nHey Darian & Shannon 👯‍️! 🌳 Çok güzel teamwork 🌻🌿 \nÇok harika ideas! 🌞✨ Tebrikler on how far you’ve come! 🎉👏 \nLet’s keep up the amazing iş 🌸, and show this proje who’s patron. 💪🌺 🌊🌍\n\n"
            # )

            if len(batch) == 4: # we don;t have any entity info
                encoder_input, decoder_input, target, mask = batch
                entity_info = None

            elif len(batch) == 5: # we have entity info
                encoder_input, decoder_input, target, mask, entity_info = batch

            encoder_input = encoder_input.to(device)
            # print("🥎len encoder input: ", encoder_input.shape)

            # print("🥎printing a sample for debugging ")
            # string = ""
            # for item in encoder_input[0]:

            #     string += semeval_train_dataset.inverse_vocab[item.item()] + " "
            # print(string)
            # print("🥎")
            # print(" ")
            decoder_input = decoder_input.to(device)
            target = target.to(device)
            mask = mask.to(device)
            if entity_info is not None:
                entity_info = entity_info.to(device)
            else:
                entity_info = None

            enc_optimizer.zero_grad()
            dec_optimizer.zero_grad()

            hidden_states, encoder_entities, encoder_inputs = encoder(encoder_input, entity_info=entity_info)
            decoder_outputs = decoder(decoder_input, hidden_states, encoder_inputs, encoder_entity_embeddings=None, entity_info=entity_info) # NOTE: the encoder returns the embeddings it used for entities
            # The decoder CAN use these embeddings by taking it in as a parameter, but it doesnt have to. If the encoder entity embeddings are not provided,
            # it will make its own entity embeddings
            # in this code, I am telling decoder to use the encoder entity embedings, but we can change this later when experimenting

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
            for batch in val_loader:
                if len(batch) == 4: # we don;t have any entity info
                    encoder_input, decoder_input, target, mask = batch
                    entity_info = None

                elif len(batch) == 5: # we have entity info
                    encoder_input, decoder_input, target, mask, entity_info = batch

                encoder_input = encoder_input.to(device)
                decoder_input = decoder_input.to(device)
                target = target.to(device)
                mask = mask.to(device)

                if entity_info is not None:
                    entity_info = entity_info.to(device)
                else:
                    entity_info = None

                hidden_states, encoder_entities, encoder_inputs = encoder(encoder_input, entity_info=entity_info)
                decoder_outputs = decoder(decoder_input, hidden_states, encoder_inputs, encoder_entity_embeddings=None, entity_info=entity_info)

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

'''# Train model
run_model(n_embd, n_head, n_layer, train_loader=semeval_train_loader,
          val_loader=semeval_val_loader, pretrain_encoder_path=pretrain_encoder_path,
          pretrain_decoder_path=pretrain_decoder_path, train_encoder_path=train_encoder_path,
          train_decoder_path=train_decoder_path, train=True)'''
