'''
Appends encoded entity to the encoded source text and then passes it to the MBart model.
'''

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import setproctitle
setproctitle.setproctitle('hiddenstate_finetuning')

from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from transformers import  Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import evaluate
from datasets import load_dataset
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
from transformers.modeling_outputs import Seq2SeqLMOutput
import json
from config import seed

torch.manual_seed(seed)
np.random.seed(seed)

data_files = {
    'train': 'smashing/smashed_train.csv',
    'validation': 'smashing/smashed_val.csv'
}

dataset = load_dataset('csv', data_files=data_files)
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-one-to-many-mmt", src_lang="en_XX")

metric = evaluate.load("accuracy")

# Convert language codes
lang_convert = {
        'ar_AE': 'ar_AR',
        'zh_TW': 'zh_CN',
        'es_ES': 'es_XX',
        'de_DE': 'de_DE',
        'fr_FR': 'fr_XX',
        'it_IT': 'it_IT',
        'ja_JP': 'ja_XX',
        'ko_KR': 'ko_KR',
        'tr_TR': 'tr_TR',
        'th_TH': 'th_TH'
    }

training_args = Seq2SeqTrainingArguments(
    output_dir="./hiddenstate",
    evaluation_strategy="epoch",  # Evaluate at the end of each epoch
    save_strategy="best",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=5e-5,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=True,  # Use FP16 if training on a GPU
    logging_dir="./logs",  # Logging directory
    logging_steps=100,  # Log every 100 steps
    push_to_hub=False,
    metric_for_best_model="eval_loss"
)

def compute_metrics(eval_pred):
    
    predictions, labels = eval_pred
    
    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)   # replaces all ignored tokens with pad token id
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)   # decodes labels and ignores special tokens
    
    # Calculate metrics for each language
    results = {}
    
    # Group predictions by language
    for lang in set(dataset['validation']['lang']):
        lang_mask = np.array(dataset['validation']['lang']) == lang
        lang_preds = np.array(decoded_preds)[lang_mask]
        lang_labels = np.array(decoded_labels)[lang_mask]
        
        # Calculate BLEU score for this language
        bleu = evaluate.load("bleu")
        bleu_score = bleu.compute(predictions=lang_preds, references=lang_labels)
        
        results[f'bleu_{lang}'] = bleu_score['bleu']
    
    # Calculate average BLEU across all languages
    results['bleu_avg'] = np.mean([v for k, v in results.items() if k.startswith('bleu_')])
    
    return results

def preprocess_function(examples):
    # Split source and entity
    source_texts = []
    entity_texts = []
    
    for text in examples["source_entity"]:
        parts = text.split("|")
        if len(parts) == 2:
            source_texts.append(parts[0] + "<" + parts[1] + ">")
            entity_texts.append(parts[1])
        else:
            source_texts.append(text)
            entity_texts.append("")
    
    # Set source language token for encoder
    tokenizer.src_lang = "en_XX"  # All our source texts are in English
    
    # Tokenize the main text and entities separately so we can get the entity embeddings
    source_inputs = tokenizer(
        source_texts, 
        padding="max_length", 
        truncation=True, 
        max_length=128,
        return_tensors="pt"
    )
    
    entity_inputs = tokenizer(
        entity_texts,
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="pt"
    )
    
    # Convert target language codes and tokenize targets for each language
    targets_list = []
    for target_text, target_lang in zip(examples["target"], examples["lang"]):
        # Set target language for this sample
        tokenizer.tgt_lang = lang_convert[target_lang]
        target_tokens = tokenizer(
            target_text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        targets_list.append(target_tokens["input_ids"])
    
    # Stack ze target tensors
    targets = torch.cat(targets_list, dim=0)

    # Keep source and entity inputs separate for this experiment
    inputs = {
        "input_ids": source_inputs["input_ids"].to(device),
        "attention_mask": source_inputs["attention_mask"].to(device),
        "entity_input_ids": entity_inputs["input_ids"].to(device),
        "entity_attention_mask": entity_inputs["attention_mask"].to(device),
        "labels": targets.to(device),
        "forced_bos_token_id": [tokenizer.lang_code_to_id[lang_convert[lang]] for lang in examples["lang"]]
    }
    
    return inputs

# custom MBart model
class MBartWithEntityModel(MBartForConditionalGeneration):
    '''
    Custom mbart implementation that appends encoded entity information 
    to the hidden states of the encoder output
    '''
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        entity_input_ids=None,
        entity_attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        if encoder_outputs is None:
            # Encode the main text
            encoder_outputs = self.model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )

            # Encode the entity separately
            if entity_input_ids is not None:
                entity_outputs = self.model.encoder(
                    input_ids=entity_input_ids,
                    attention_mask=entity_attention_mask,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=True,
                )
                
                # Append entity hidden states to encoder outputs
                # Take mean of entity hidden states across sequence length
                entity_hidden = torch.mean(entity_outputs.last_hidden_state, dim=1, keepdim=True)
                
                # Expand entity hidden states to match batch size
                # !!!! MAY DAY MAY DAY DOES THIS MAKE SENSE CHAT? 
                entity_hidden = entity_hidden.expand(-1, encoder_outputs.last_hidden_state.size(1), -1)
                
                # Concatenate entity hidden states to encoder outputs
                encoder_outputs.last_hidden_state = torch.cat(
                    [encoder_outputs.last_hidden_state, entity_hidden], dim=-1
                )
                
                # Update attention mask to account for appended states
                attention_mask = attention_mask.clone()

        # Get decoder outputs with modified encoder hidden states
        outputs = self.model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        sequence_output = outputs.last_hidden_state

        # Project to vocabulary
        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.hidden_states,
            decoder_attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

def generate_and_save_translations(text, target_langs, output_dir="append_hidden_states"):
    """
    Generate translations and save them to separate files by language
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    source_parts = text.split("|")
    source_text = source_parts[0]
    entity = source_parts[1] if len(source_parts) > 1 else ""
    
    # Prepare inputs
    source_inputs = tokenizer(source_text, return_tensors="pt", padding=True)
    entity_inputs = tokenizer(entity, return_tensors="pt", padding=True)
    
    for target_lang in target_langs:
        output_file = os.path.join(output_dir, f"{target_lang}.jsonl")
        
        # Set target language
        forced_bos_token_id = tokenizer.lang_code_to_id[lang_convert[target_lang]]
        
        # Generate translation
        generated_tokens = model.generate(
            input_ids=source_inputs["input_ids"].to(model.device),
            attention_mask=source_inputs["attention_mask"].to(model.device),
            entity_input_ids=entity_inputs["input_ids"].to(model.device),
            entity_attention_mask=entity_inputs["attention_mask"].to(model.device),
            forced_bos_token_id=forced_bos_token_id,
            max_length=128,
            num_beams=5,
            length_penalty=1.0,
        )
        
        translated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        
        # Create output dictionary
        output = {
            "source": source_text,
            "entity": entity,
            "translation": translated_text
        }
        
        # Append to language-specific file
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(output, ensure_ascii=False) + '\n')
        
        print(f"Saved translation for {target_lang}: {translated_text}")


if __name__ == '__main__':  

    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    # initialize pretrained model
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-one-to-many-mmt").to(device)

    # initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
    )

    trainer.train()
    trainer.save_model("./hiddenstate/model")
    tokenizer.save_pretrained("./hiddenstate/config")

    # Test text
    text = "Has Bernie Sanders ever been president of the United States?|بيرني ساندرز"

    # Generate and save translations for all supported languages
    target_langs = ["ar_AE", "es_ES", "fr_FR", "de_DE", "it_IT", "ja_JP", "ko_KR", "tr_TR", "th_TH"]
    generate_and_save_translations(text, target_langs)

