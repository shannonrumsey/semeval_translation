import pandas
from datasets import load_dataset
import pandas as pd
from datasets import concatenate_datasets
import os
import json
import sys
import sentencepiece as spm
import numpy as np
from numpy.random import Generator, PCG64

# summary of changes 12/21:
# pretrain now returns a dictionary of dfs for each language. This is needed to allow us to split after bpe in order to preserve sentence lengths

def get_data(lang):
    """
    Gets quesiton data from Hugging Face datasets.

    Args:
        lang (list): A list of Hugging Face SQuAD datasets (in same languages as fine-tuning dataset).
                     Examples:
                        - Single-component strings: "squad_it", "rajpurkar/squad"
                        - Two-component lists: ["google/xquad", "xquad.ar"], where the second element is the config
    Returns:
        A dictionary of sentence dfs for each language ending in </s>

    Notes:
        - AR, DE, ES have two components (["facebook/mlqa", ""mlqa-translate-train.lang_code"])
        - IT is small so the train and test split are combined
    """
    dfs = {}

    for l in lang:


        # AR, DE, and ES have two components and only val data
        if len(l) == 2:
            df = pd.DataFrame()
            data = load_dataset(l[0], l[1])["train"]
            # data = [example["question"] for example in data]
            lang_code = l[1].split(".")[-1]
            data = [example["question"] + " </s>" for example in data]
            dfs[lang_code] = pd.DataFrame({"text": data})
            # print(len(data))



        # IT has combined train and test
        elif l == "squad_it":
            data = load_dataset(l)
            data = concatenate_datasets([data["train"], data["test"]])
            # data = [example["question"] for example in data]
            data = [example["question"] + " </s>" for example in data]
            dfs[lang_code] = pd.DataFrame({"text": data})
            # print(len(data))


        else:
            data = load_dataset(l)["train"]
            # data = [example["question"] for example in data]
            if "fr" in l:
                lang_code = "fr"
            elif "JaQuAD" in l:
                lang_code = "ja"
            else:
                lang_code = "en"

            data = [example["question"] + " </s>" for example in data]
            dfs[lang_code] = pd.DataFrame({"text": data})
            # print(len(data))

        #df = pd.concat([df, pd.DataFrame(data, columns=["text"])], ignore_index=True)

    return dfs

lang = ["qwant/squad_fr", "squad_it", "rajpurkar/squad", "SkelterLabsInc/JaQuAD", ["facebook/mlqa", "mlqa-translate-train.ar"], ["facebook/mlqa", "mlqa-translate-train.de"], ["facebook/mlqa", "mlqa-translate-train.es"]]
pretrain = get_data(lang)
# ^ pretrain now is a dict that contains seperate data frames for each language

# just printing out the head of each df to make sure our results look normal
for key, value in pretrain.items():
    print(key)
    print(value.head())

# WARNING, WE ARE NOW XITED THE SHANNON SECTION AND ENTERING THE DARIAN SECTION
# PREPARE FOR MUCH LESS PRETTY CODE XD
# ==== we are loading the semeval data to do bpe on everything together ====

# base dir will need to be edited if this is run on a different computer
base_dir = os.path.join(os.path.dirname(__file__), "data/semeval_train")

def get_semeval_data(base_dir, for_bpe=False):
    # for_bpe allows user to specify what format the data should be in
    # for_bpe will return dfs with a single column 'text' so that it can be combined with the pretrain data to run bpe

    # dictionary to store loaded data for each language
    semeval_train = {}

    # expected format: train -> [ar -> train.jsonl, de -> train.jsonl...]
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)

        # check if the path is a language folder
        if os.path.isdir(folder_path):
            jsonl_file_path = os.path.join(folder_path, "train.jsonl")

            if os.path.isfile(jsonl_file_path):
                with open(jsonl_file_path, "r", encoding="utf-8") as jsonl_file:

                    if for_bpe:
                        data_target = [json.loads(line)["target"] for line in jsonl_file if "target" in json.loads(line)] # only gets the line with the translation
                        data_source = [json.loads(line)["source"] for line in jsonl_file if
                                       "source" in json.loads(line)]
                        combined_data = data_target + data_source

                        df = pd.DataFrame({"text": combined_data})

                    else:
                        data = [json.loads(line) for line in jsonl_file]
                        df = pd.DataFrame(data)

                    semeval_train[folder_name] = df

    return semeval_train

# data_by_language now has data for each language
semeval_train = get_semeval_data(base_dir, for_bpe=True)

def get_text_file_for_sentencepiece():
    # creating a text corpus for BPE
    # this is needed to ensure that the pretrain data and the training data has the same vocabulary

    big_df = pd.DataFrame()
    for key in semeval_train:
        df = semeval_train[key]

        big_df = pd.concat([big_df, df], ignore_index=True)
    for key, value in pretrain.items():

        big_df = pd.concat([big_df, value], ignore_index=True)


    corpus = "\n".join(big_df["text"].dropna())
    with open("corpus_for_bpe.txt", "w", encoding="utf-8") as f:
        f.write(corpus)

    return "corpus_for_bpe.txt"


corpus = get_text_file_for_sentencepiece()


# BPE
# Removing dummy prefix correctly formats language codes
spm.SentencePieceTrainer.train(input=corpus, model_prefix="tokenizer_combined", vocab_size=30000, add_dummy_prefix=False,
                               character_coverage=0.9995, model_type="bpe",
                               user_defined_symbols=["</s>", "<es>", "<fr>", "<it>", "<de>", "<ar>", "<ja>", "<en>"])

def apply_bpe_tokenizer(df, column_name):
    sp = spm.SentencePieceProcessor(model_file="tokenizer_combined.model")
    df[column_name] = df[column_name].apply(lambda text: sp.encode(text, out_type=str))
    return df


def get_chunks_from_corpuses(dataframes, chunk_size=35):

    """Inputs:
            chunk size: the size of chunks of continuos text, defaulted at 35, which was used in pretraining the previous model and led to good results
            dataframes: a dictionary of dataframes for each language

        outputs:
            sampled_dfs: a dfs of text for each language where each row has the same sequence length"""
    new_dfs = {}
    lens = {}
    for key, df in dataframes.items():
        sub_word_df = apply_bpe_tokenizer(df, "text")
        sub_word_df["text"] = sub_word_df["text"].apply(lambda row: " ".join(row))
        print(sub_word_df.head())
        corpus = " ".join(sub_word_df["text"].dropna()) # join all the text in the df into one corpus so that chunking can occur


        chunks = []

        # a loop for processing the corpus into chunks and appending them to the df
        not_processed = corpus.split(" ") # a list of all the words in the corpus in order
        while len(not_processed) > chunk_size:
            current_chunk = not_processed[:chunk_size]
            if key[0] == "<":
                chunks.append([key] + current_chunk) # for some reason, some of the keys are in the form "en" and others are in the form "<en>"
            else:
                chunks.append(["<" + key + ">"] + current_chunk)

            # note: to reduce noise, we allow some repetiton by making sure the chunks always start at the beginning of a sentence
            # example: if text is:
            # "Toads have different distinctive features than what typically characterizes a frog. Often toads have drier, bumpier “warty” skin and prefer drier habitats. They usually have shorter hind limbs and rounder stouter bodies than most typical frogs. Toads have poison glands in their skin to keep predators from eating them and oftentimes produce a funny smell when handled."
            # and chunk size is 15, then output will be:
            # 1. Toads have different distinctive features than what typically characterizes a frog. Often toads have drier
            # 2. Often toads have drier, bumpier “warty” skin and prefer drier habitats. They usually have shorter
            # 3. They usually have shorter hind limbs and rounder stouter bodies than most typical frogs. Toads

            # this allows for some repetition, but is less noisey for the model
            # this is a variant to the sliding window approach used in training most state of the art models


            if "</s>" in current_chunk:
                reversed_index = list(reversed(current_chunk)).index("</s>")  # reverse() followed by index() finds the last index of "</s>"
                index = len(current_chunk) - reversed_index - 1 # unreversing it
                not_processed = not_processed[index + 1:]
            else:
                print("No </s> found, using chunk size.")
                # if no "</s>" is found, just return the next chunk
                not_processed = not_processed[chunk_size:]




        chunked_df = pd.DataFrame({"text": chunks})
        new_dfs[key] = chunked_df

        lens[key] = len(chunks) # keep track of the lowest occuring class inorder to avoid class inbalances
    print(lens)
    min_len = min(lens.values())

    sampled_dfs = {key: df.sample(n=min_len, random_state=12).reset_index(drop=True) for key, df in new_dfs.items()}

    return sampled_dfs




pretrain = get_chunks_from_corpuses(pretrain, 35)
print("after_chunking!!!")
for key, df in pretrain.items():
    print(key)
    print(df.head())
    print("size: ", len(df))

def noise(row, rng):
    """
    Randomly masks words in a sentence.
    Args:
        row (BPE list): A list of a sentence that has been encoded using BPE.
    Returns:
        noisy_row (BPE list): A list that has noise introduced via random masking (encoder input).
        text (BPE list): A list containing the original sequence before masking (decoder input).
        shifted (BPE list): A list of the original text, shifted over one token to the right (expected model generation).

    Notes:
        - The span length sampled from the Poisson distribution cannot be greater than the length of the sentence
        - If the span length is greater, just change it to be the same as the text length
        - The language tag should never be masked
        - [MASK] token is not in the BPE vocab because the model should predict a replacement for it!
    """
    span_len = row["span_len"]
    text = row["text"]

    if span_len > len(text) - 1:
        span_len = len(text) - 1

    # Randomly select index to be masked, repeat this span_len times
    to_corrupt = rng.choice(text[1:], size=span_len, replace=False)
    
    noisy_row = ["[MASK]" if word in to_corrupt else word for word in text]

    # Add lang code at the end to make shifted the same length as text
    shifted = text[1:] + list(text[0])
    
    # move text over an indices to be decoder input
    return noisy_row, text, shifted

    
rng = Generator(PCG64())
# Randomly select span length (number of words to be masked)
masked_dfs = {}
output_folder = "processed_pretrain_dfs"
os.makedirs(output_folder, exist_ok=True)

# saving all individual dfs and a merged df in a folder called processed_pretrain_dfs

all_dfs = []
for lang_key, pretrain_df in pretrain.items():
    span_len = rng.poisson(3.5, size=len(pretrain_df["text"]))
    df = pd.concat([pd.DataFrame(span_len, columns=["span_len"]), pretrain_df], axis=1)
    df[["encoder_input", "decoder_input", "decoder_output"]] = df.apply(lambda row: pd.Series(noise(row, rng)), axis=1)
    df.drop(["span_len", "text"], axis=1, inplace=True)
    masked_dfs[lang_key] = df
    all_dfs.append(df)

for lang_key, masked_df in masked_dfs.items():
    file_path = os.path.join(output_folder, f"{lang_key}_masked_data.csv")
    masked_df.to_csv(file_path, index=False)


merged_df = pd.concat(all_dfs, ignore_index=True)


merged_file_path = os.path.join(output_folder, "all_masked_data.csv")
merged_df.to_csv(merged_file_path, index=False)




