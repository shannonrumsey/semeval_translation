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

# Note: entities_in_self_attn is a flag ONLY for our fourth experiment. It just adds an if-else statement in the decoder to not use entities during self attention
vocab_size = len(pretrain_dataset.vocab)
# in order to set the proper size for max seq length for our positional embeddings
def find_max_sequence_length(dataset, entity= False): # if entity == True, it will return the max entitry length
    if entity:
        if dataset.entity_ids != None:

            longest_entity = max(len(ids) for ids in dataset.entity_ids)
            return longest_entity
    else:
        
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

if entity_len_train and entity_len_val:
    print("entities in val and train found")
    entity_len = max(entity_len_train, entity_len_val)
else:
    
    if entity_len_train:
        print("only entities in train")
        entity_len = entity_len_train
    elif entity_len_val:
        print("only entities in val")
        entity_len = entity_len_val

    else:
        print("no entities anywhere")
        entity_len = 10


n_embd = 512
n_head = 8
n_layer = 4

def check_gradients(model):
    max_norm = 0
    min_norm = float('inf')
    avg_norm = 0
    num_params = 0

    for param in model.parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            max_norm = max(max_norm, grad_norm)
            min_norm = min(min_norm, grad_norm)
            avg_norm += grad_norm
            num_params += 1

    avg_norm /= max(num_params, 1)

    print(f"Gradient Stats - Max: {max_norm:.4f}, Min: {min_norm:.4f}, Avg: {avg_norm:.4f}")

    # Warnings for vanishing or exploding gradients
    if max_norm > 5:
        print("⚠️ Warning: Possible exploding gradients detected (Max norm > 5)!")
    if min_norm < 1e-6:
        print("⚠️ Warning: Possible vanishing gradients detected (Min norm < 1e-6)!")


def print_gradient_stats(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"{name} | Gradient Norm: {grad_norm:.6f}")

def run_model(n_embd, n_head, n_layer, train_loader, val_loader, pretrain_encoder_path, pretrain_decoder_path, train_encoder_path=None, train_decoder_path=None, train=False):
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
    
    num_epoch = 30
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

            hidden_states, encoder_entity_embeddings, encoder_inputs = encoder(encoder_input, entity_info=None) # encoder inputs used to create pad mask for cross attention
            #print("hidden_states : ", hidden_states)
            
            decoder_outputs = decoder(decoder_input, hidden_states, encoder_inputs, encoder_entity_embeddings=None, entity_info=None, use_encoders_entities=False, entities_in_self_attn=True) # NOTE: the encoder returns the embeddings it used for entities

            #print("decoder_outputs: ", decoder_outputs)
            # The decoder CAN use these embeddings by taking it in as a parameter, but it doesnt have to. If the encoder entity embeddings are not provided,
            # it will make its own entity embeddings
            # in this code, I am telling decoder to use the encoder entity embedings, but we can change this later when experimenting
            loss = loss_fn(decoder_outputs.view(-1, vocab_size), target.view(-1))

            loss.backward()

            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1)

            # print("Encoder gradients")
            # print_gradient_stats(encoder)
            # check_gradients(encoder)

            # print("Decoder gradients")
            # print_gradient_stats(decoder)
            # check_gradients(decoder)

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

                hidden_states, encoder_entity_embeddings, encoder_inputs = encoder(encoder_input, entity_info=None)
                #print("hidden states: ", hidden_states)
                decoder_outputs = decoder(decoder_input, hidden_states, encoder_inputs, encoder_entity_embeddings=None, entity_info=None, use_encoders_entities=False, entities_in_self_attn=True)
                #print("decoder_outputs: ", decoder_outputs)
                loss = loss_fn(decoder_outputs.view(-1, vocab_size), target.view(-1))
                val_loss += loss.item()

            total_val_loss = val_loss/ len(val_loader)
            print(f"\n\n🥭🍏🍋🍎🍒🍆🥬🫐 Pretrain Validation Loss on Epoch {epoch}: {total_val_loss}")
            
            # Save model with lowest loss
            if prev_loss is None or total_val_loss < prev_loss:
                print("LOWEST LOSS: SAVING MODEL")
                torch.save(encoder.state_dict(), encoder_path)
                torch.save(decoder.state_dict(), decoder_path)
                prev_loss = total_val_loss

        target_tokens = target.tolist()
        encoder_tokens = encoder_input.tolist()
        EXCLUDE_TOKENS = {"_", "</s>"}  
        exclude_indices = {idx for idx, token in pretrain_dataset.inverse_vocab.items() if token in EXCLUDE_TOKENS}

        masked_outputs = decoder_outputs.clone()

        for idx in exclude_indices:
            masked_outputs[:, :, idx] = float('-inf')

  
        pred_tokens = masked_outputs.argmax(dim=-1).tolist()
        _, top_indices = masked_outputs.topk(2, dim=-1)
        second_max_indices = top_indices[:, :, 1]
        second_pred_tokens = second_max_indices.tolist()


        

        print("\n\n🍋🍋 with highest 🍋🍋")
        for i in range(min(5, len(pred_tokens))):
            pred_sentence = " ".join([
                pretrain_dataset.inverse_vocab[token]
                for token in pred_tokens[i] if token in pretrain_dataset.inverse_vocab
            ])

            target_sentence = " ".join([
                pretrain_dataset.inverse_vocab[token]
                for token in target_tokens[i] if token in pretrain_dataset.inverse_vocab
            ])
            encoder_sentence = " ".join([pretrain_dataset.inverse_vocab[token] for token in encoder_tokens[i] if token in pretrain_dataset.inverse_vocab])

            print(f"predicted {i+1}: {pred_sentence}")
            print(f"target {i+1}: {target_sentence}\n")
            print(f"encoder {i+1}: {encoder_sentence}\n")
        mask_token_idx = next(idx for idx, token in pretrain_dataset.inverse_vocab.items() if token == "[MASK]")
        masked_positions = [
            [i for i, token in enumerate(seq) if token == mask_token_idx]
            for seq in encoder_tokens
        ]


        correct, total = 0, 0

        for i, mask_idxs in enumerate(masked_positions):
            for idx in mask_idxs:
                if pred_tokens[i][idx] == target_tokens[i][idx]:  
                    correct += 1
                total += 1

        
        accuracy = correct / total if total > 0 else 0
        print(f"\n🎯 Accuracy on masked tokens: {accuracy:.4f}")

        
pretrain_encoder_path = os.path.join(os.path.dirname(__file__), "trained_models/pretrain_encoder_model")
pretrain_decoder_path = os.path.join(os.path.dirname(__file__), "trained_models/pretrain_decoder_model")
train_encoder_path = os.path.join(os.path.dirname(__file__), "trained_models/train_encoder_model")
train_decoder_path = os.path.join(os.path.dirname(__file__), "trained_models/train_decoder_model")

# Pretrain model
run_model(n_embd, n_head, n_layer, train_loader=pretrain_train_loader,
        val_loader=pretrain_val_loader, pretrain_encoder_path=pretrain_encoder_path,
        pretrain_decoder_path=pretrain_decoder_path, train=False)

# Train model
'''run_model(n_embd, n_head, n_layer, train_loader=semeval_train_loader,
          val_loader=semeval_val_loader, pretrain_encoder_path=pretrain_encoder_path,
          pretrain_decoder_path=pretrain_decoder_path, train_encoder_path=train_encoder_path,
          train_decoder_path=train_decoder_path, train=True)
'''