from datasets import load_dataset
import pandas as pd
from datasets import concatenate_datasets
import os
import json
import sys
import sentencepiece as spm
import numpy as np
from numpy.random import Generator, PCG64

# rng = Generator(PCG64())
# print(rng.poisson(3.5))

# sys.exit(0)
def get_data(lang):
    """
    Gets quesiton data from Hugging Face datasets.

    Args:
        lang (list): A list of Hugging Face SQuAD datasets (in same languages as fine-tuning dataset).
                     Examples:
                        - Single-component strings: "squad_it", "rajpurkar/squad"
                        - Two-component lists: ["google/xquad", "xquad.ar"], where the second element is the config
    Returns:
        pd.DataFrame: A DataFrame containing all the questions from the specified datasets


    Notes:
        - AR, DE, ES have two components (["google/xquad", "xquad.lang"]) and only has a val split
        - IT is small so the train and test split are combined
    """
    df = pd.DataFrame()

    for l in lang:
        corpus = ""

        # AR, DE, and ES have two components and only val data
        if len(l) == 2:
            data = load_dataset(l[0], l[1])["train"]
            data = [example["question"] for example in data]
            # print(len(data))



        # IT has combined train and test
        elif l == "squad_it":
            data = load_dataset(l)
            data = concatenate_datasets([data["train"], data["test"]])
            data = [example["question"] for example in data]
            # print(len(data))


        else:
            data = load_dataset(l)["train"]
            data = [example["question"] for example in data]
            # print(len(data))


        df = pd.concat([df, pd.DataFrame(data, columns=["text"])], ignore_index=True)

    return df

lang = ["qwant/squad_fr", "squad_it", "rajpurkar/squad", "SkelterLabsInc/JaQuAD", ["facebook/mlqa", "mlqa-translate-train.ar"], ["facebook/mlqa", "mlqa-translate-train.de"], ["facebook/mlqa", "mlqa-translate-train.es"]]
pretrain = get_data(lang)



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
    big_df = pd.concat([big_df, pretrain], ignore_index=True)


    corpus = " ".join(big_df["text"].dropna())
    with open("corpus_for_bpe.txt", "w", encoding="utf-8") as f:
        f.write(corpus)

    return "corpus_for_bpe.txt"


corpus = get_text_file_for_sentencepiece()

# BPE
def apply_bpe_tokenizer(df, column_name):
    try:
        spm.SentencePieceTrainer.train(input=corpus, model_prefix="tokenizer_combined", vocab_size=30000, character_coverage=0.9995, model_type="bpe")
        sp = spm.SentencePieceProcessor(model_file="tokenizer_combined.model")
        df[column_name] = df[column_name].apply(lambda text: sp.encode(text, out_type=str))

    except RuntimeError as e:
        if "Vocabulary size too high" in str(e):
            max_vocab_size = int(str(e).split("value <= ")[1].strip("."))
            print(f"Changing max vocab size to {max_vocab_size}")
            spm.SentencePieceTrainer.train(input=corpus, model_prefix="tokenizer_combined", vocab_size=max_vocab_size, character_coverage=0.9995, model_type="bpe")
            sp = spm.SentencePieceProcessor(model_file="tokenizer_combined.model")
            df[column_name] = df[column_name].apply(lambda text: sp.encode(text, out_type=str))

    return df

pretrain = apply_bpe_tokenizer(pretrain, "text")

def noise(row):
    """
    Randomly masks words in a sentence.
    Args:
        row (BPE list): A list of a sentence that has been encoded using BPE.
    Returns:
        noisy_row (BPE list): A list that has noise introduced via random masking.
    Notes:
        - The span length sampled from the Poisson distribution must be shorter than the sentence length.
        - 
    """
    # Randomly select span length (number of words to be masked)
  
    span_len = rng.poisson(3.5)

    if span_len > len(row):
        span_len = 0

    print(row, span_len)
    # Randomly select index to be masked, repeat this span_len times
    indices = rng.choice(row, size=span_len, replace=False)
    
    noisy_row = ["[MASK]" if ind in indices else word for ind, word in enumerate(row)]
    print(span_len, noisy_row)

    return noisy_row

    
rng = Generator(PCG64())
pretrain.apply(lambda row: noise(row), axis=1)
