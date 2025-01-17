import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import os
import json
import sentencepiece as spm
import random
import ast
from sklearn.model_selection import train_test_split
import sys

# define dataset
print("defining the dataset")
class TranslationDataset(Dataset):
    """
    Loads the data from the CSV files generated in pretrain.py
    """
    def __init__(self, vocab = None):
        if vocab != None:
            self.vocab = vocab
        else:
            self.vocab = {"<PAD>": 0, "<unk>": 1, "[ent_info]": 2, "->": 3}  # special symbols which may not be in the data but need to be included

        self.inverse_vocab = None
        self.corpus_encoder_ids = []
        self.corpus_decoder_ids = []
        self.corpus_target_ids = []
        self.corpus_y_mask = [] # In the case of pretrain, this will be a dummy variable (all 1s, indicating no padding)
        self.entity_ids = None

    def make_vocab(self, pretrain_data, train_data, entity_data = None):
        """
        Args:
            - pretrain_data (DataFrame): contains encoder input, decoder input, and decoder output (expected output).
            - train_data (list of DataFrames): each DataFrame is associated with a language.
            - entity_data (list of dataFrames): each DataFrame is associated with a language. DataFrames contain entity information
                                        with column "source" being the english translation and column "target" being the forgien translation
        Note:
            - Both pretrain and train data will be used in the BPE vocabulary.
            - train_data is a list of DataFrames because a combined one is too large for Github.
        """

        # add tokens from pretrain data to vocab
        for df in pretrain_data:
            for row in df["encoder_input"]:
                if isinstance(row, list):
                    pass
                else: # in some of the dfs, the lists got saved as strings
                    try:
                        row = ast.literal_eval(row)
                        if isinstance(row, list):
                            pass
                            #print("Successfully converted to list:", row)
                        else:
                            print(f"row is not a list, it is of type {type(row)}: {row}")
                    except (ValueError, SyntaxError):
                        print("Error: The string could not be interpreted as a list.")
                for token in row:

                    if token not in self.vocab:
                        self.vocab[token] = len(self.vocab)

            for row in df["decoder_input"]: # most of these will probably already be in the input, but the masked ones might not be
                if isinstance(row, list):
                    pass
                else:  # in some of the dfs, the lists got saved as strings
                    try:
                        row = ast.literal_eval(row)
                        if isinstance(row, list):
                            pass
                            #print("Successfully converted to list:", row)
                        else:
                            print(f"row is not a list, it is of type {type(row)}: {row}")
                    except (ValueError, SyntaxError):
                        print("Error: The string could not be interpreted as a list.")
                for token in row:

                    if token not in self.vocab:
                        self.vocab[token] = len(self.vocab)

            # we shouldnt need to do this on the decoder outputs because they will be the same

        # add tokens in train data to vocab as well
        for df in train_data:
            if isinstance(df, pd.DataFrame):
                print("train is a df")
            else:
                print("train is NOT a df")


            for row in df["source"]:
                if isinstance(row, list):
                    pass
                else:  # in some of the dfs, the lists got saved as strings
                    try:
                        row = ast.literal_eval(row)
                        if isinstance(row, list):
                            pass
                            #print("Successfully converted to list:", row)
                        else:
                            print(f"row is not a list, it is of type {type(row)}: {row}")
                    except (ValueError, SyntaxError):
                        print("Error: The string could not be interpreted as a list.")

                for token in row:
                    if token not in self.vocab:
                        self.vocab[token] = len(self.vocab)
            for row in df["target"]:
                for token in row:
                    if token not in self.vocab:
                        self.vocab[token] = len(self.vocab)

        ## adds entity data to vocab in case any is missing (most of it will probably already be present
        if entity_data:
            for df in entity_data:
                if isinstance(df, pd.DataFrame):
                    print("entity_data is a df")
                else:
                    print(df)
                    print(type(df))

                    print("entity_data is NOT a df")

                for row in df["target"]:

                    for token in row:
                        if token not in self.vocab:
                            self.vocab[token] = len(self.vocab)

                for row in df["source"]: # zero is source
                    for token in row:
                        if token not in self.vocab:
                            self.vocab[token] = len(self.vocab)


        self.inverse_vocab = {index: token for token, index in self.vocab.items()}

    def load_vocab(self, vocab): # this method will be used for val amd test sets to ensure they have the same vocab
        self.vocab = vocab
        self.inverse_vocab = {index: token for token, index in self.vocab.items()}

    def encode_pretrain(self, pretrain_data):
        # pretrain_data expected to be a list of dfs with the columns encoder input, decoder input, decoder output

        if self.vocab is None:
            raise ValueError("🚩No vocab found 🚩. Please build vocab using 'make_vocab()' and try again.")
        for df in pretrain_data:
            for row in df["encoder_input"]:
                if isinstance(row, str):  # some of the lists were improperly encoded as strings
                    try:
                        row = ast.literal_eval(row)  # force the string rows to be lists
                        if not isinstance(row, list):
                            print(f"Error: The string could not be interpreted as a list: {row}")
                            continue
                    except (ValueError, SyntaxError):
                        print(f"Error: The string could not be interpreted as a list: {row}")
                        continue
                encoder_ids = [self.vocab.get(token, self.vocab['<unk>']) for token in row]

            for row in df["decoder_input"]:
                if isinstance(row, str):  # some of the lists were improperly encoded as strings
                    try:
                        row = ast.literal_eval(row)  # force the string rows to be lists
                        if not isinstance(row, list):
                            print(f"Error: The string could not be interpreted as a list: {row}")
                            continue
                    except (ValueError, SyntaxError):
                        print(f"Error: The string could not be interpreted as a list: {row}")
                        continue
                decoder_ids = [self.vocab.get(token, self.vocab['<unk>']) for token in row]

            for row in df["decoder_output"]:
                if isinstance(row, str):  # some of the lists were improperly encoded as strings
                    try:
                        row = ast.literal_eval(row)  # force the string rows to be lists
                        if not isinstance(row, list):
                            print(f"Error: The string could not be interpreted as a list: {row}")
                            continue
                    except (ValueError, SyntaxError):
                        print(f"Error: The string could not be interpreted as a list: {row}")
                        continue
                target_ids = [self.vocab.get(token, self.vocab['<unk>']) for token in row]

            self.corpus_encoder_ids.append(torch.tensor(encoder_ids))
            self.corpus_decoder_ids.append(torch.tensor(decoder_ids))
            self.corpus_target_ids.append(torch.tensor(target_ids))
            dummy_mask = torch.ones(len(target_ids))
            # we will need to have masks for the actual task to only take the loss on the translation part.
            # For pretrain we will just use a dummy mask of all 1s so that we dont need to change the code greatly
            self.corpus_y_mask.append(dummy_mask)

        # paired data just allows us to shuffle the data from different languages without losing information
        paired_data = list(
            zip(self.corpus_encoder_ids, self.corpus_decoder_ids, self.corpus_target_ids, self.corpus_y_mask))
        random.shuffle(paired_data)
        self.corpus_encoder_ids, self.corpus_decoder_ids, self.corpus_target_ids, self.corpus_y_mask = zip(*paired_data)

        # convert from iter data type to list data type
        self.corpus_encoder_ids, self.corpus_decoder_ids, self.corpus_target_ids, self.corpus_y_mask = list(
        self.corpus_encoder_ids), list(self.corpus_decoder_ids), list(self.corpus_target_ids), list(
        self.corpus_y_mask)

    def encode_semeval(self, data, entity_data = None): # for training, val, amd test semeval data
        """
        Args:
            data (list of DataFrames): Similar to the encode_pretrain inputs where the SemEval data is split up into a list of DataFrames
                                        by language.
            entity_data (list of dataFrames): each DataFrame is associated with a language. DataFrames contain entity information
                                        with column "source" being the english translation and column "target" being the forgien translation
        Outputs: 
                se_corpus_encoder_ids (list of lists): encoder inputs (English sentence)
                se_corpus_decoder_ids (list of lists): decoder inputs (translated sentence)
                se_corpus_target_ids (list of lists): expected decoder outputs (translated sentence shifted one token to the right)
                se_corpus_y_mask (list of lists): masks applied to the corpus_target_ids for training and inference
        Notes:
            - The resulting lists combines all of the languages together and shuffles them to prevent patterns like en en..., es es..., it it...
        """
        if self.vocab is None:
            raise ValueError("🚩No vocab found 🚩. Please build vocab using 'make_vocab()' and try again.")
        
        for df in data:
            source = df["source"]
            target = df["target"]
            lang = df["target_locale"]

            for src, trg, l in zip(source, target, lang):
                encoder_input = [l] + src + ["</s>"]
                encoder_ids = [self.vocab.get(token, self.vocab['<unk>']) for token in encoder_input]

                decoder_input = trg + ["</s>"]
                decoder_ids = [self.vocab.get(token, self.vocab['<unk>']) for token in decoder_input]

                target = decoder_input[1:] + [l]  # Add random token to end to ensure deocder input length = target length
                target_ids = [self.vocab.get(token, self.vocab['<unk>']) for token in target]

        
            self.corpus_encoder_ids.append(torch.tensor(encoder_ids))
            self.corpus_decoder_ids.append(torch.tensor(decoder_ids))
            self.corpus_target_ids.append(torch.tensor(target_ids))
            dummy_mask = torch.ones(len(target_ids))
            self.corpus_y_mask.append(dummy_mask)

        if entity_data:
            self.entity_ids = []

            for df in entity_data:
                source = df["source"]
                target = df["target"]

                for s, t in zip(source, target):
                    s_tokens = [self.vocab.get(token, self.vocab['<unk>']) for token in s]
                    t_tokens = [self.vocab.get(token, self.vocab['<unk>']) for token in t]

                    new_sentence = [self.vocab["[ent_info]"]] + s_tokens + [self.vocab["->"]] + t_tokens
                    print(new_sentence)
                    self.entity_ids.append(torch.tensor(new_sentence))

        # Shuffle data from all languages

        if self.entity_ids is not None:
            paired_data = list(
                zip(self.corpus_encoder_ids, self.corpus_decoder_ids, self.corpus_target_ids, self.corpus_y_mask,
                    self.entity_ids)
            )
            random.shuffle(paired_data)
            self.corpus_encoder_ids, self.corpus_decoder_ids, self.corpus_target_ids, self.corpus_y_mask, self.entity_ids = zip(
                *paired_data)
        else:
            paired_data = list(
                zip(self.corpus_encoder_ids, self.corpus_decoder_ids, self.corpus_target_ids, self.corpus_y_mask)
            )
            random.shuffle(paired_data)
            self.corpus_encoder_ids, self.corpus_decoder_ids, self.corpus_target_ids, self.corpus_y_mask = zip(
                *paired_data)

        # Convert from iter data type to list data type
        self.corpus_encoder_ids, self.corpus_decoder_ids, self.corpus_target_ids, self.corpus_y_mask = list(
        self.corpus_encoder_ids), list(self.corpus_decoder_ids), list(self.corpus_target_ids), list(
        self.corpus_y_mask)

        if self.entity_ids:
            self.entity_ids = list(self.entity_ids)

    
    def make_sure_everythings_alligned_properly(self, train = False):
        print("making sure everything looks good in the dataset")
        random_index = random.randint(0, len(self.corpus_encoder_ids) - 1) # to get a random sample from the data
        print("fetching data at random index: ", random_index)
        encoder = self.corpus_encoder_ids[random_index]
        decoder = self.corpus_decoder_ids[random_index]
        target = self.corpus_target_ids[random_index]
        mask = self.corpus_y_mask[random_index]
        if train is False:
            print("\n====== CHECKING LENGTHS =====")
            print("encoder_ids: ", len(encoder), "decoder_ids: ", len(decoder), "target_ids: ", len(target), "mask_ids: ", len(mask))
            if len(encoder) != len(decoder):
                print("\n====== MISMATCH FOUND ======")
                print("encoder and decoder lengths do not match!")
                # Print the elements that do not align
                print("\n==== MISMATCHED WORDS ====")
                for e, d in zip(encoder, decoder):
                    real_encoder = self.inverse_vocab.get(e.item(), "<unk>")
                    real_decoder = self.inverse_vocab.get(d.item(), "<unk>")
                    if real_encoder != real_decoder:
                        print(f"Encoder: {real_encoder} | Decoder: {real_decoder}")

        #print("\n======= numerical values ======")
        #print("\nencoder_ids: ", encoder, "\ndecoder_ids: ", decoder, "\ntarget_ids: ", target, "\nmask_ids: ", mask)

        real_encoder = [self.inverse_vocab.get(token.item(),  "<unk>") for token in encoder]
        real_decoder = [self.inverse_vocab.get(token.item(), "<unk>") for token in decoder]
        real_target = [self.inverse_vocab.get(token.item(), "<unk>") for token in target]

        print("\n======= decoded values =======")
        print("\nencoder: ", real_encoder, "\ndecoder: ", real_decoder, "\ntarget: ", real_target,
                "\nmask: ", mask)

    def __len__(self):
        return len(self.corpus_encoder_ids)

    def __getitem__(self, idx):
        if self.entity_ids:
            return self.corpus_encoder_ids[idx], self.corpus_decoder_ids[idx], self.corpus_target_ids[idx], \
            self.corpus_y_mask[idx], self.entity_ids[idx]

        else:
            return self.corpus_encoder_ids[idx], self.corpus_decoder_ids[idx], self.corpus_target_ids[idx], self.corpus_y_mask[idx]


pretrain_list = []
folder_path = os.path.abspath("data/processed_pretrain")

for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        pretrain_list.append(df)  
        

def get_semeval_train(just_get_lines = False): # knowing the lines will be used to check if the entities line up with the train dfs
    semeval_train = []
    rows_per_df = [] # once again, this will help us detect misallignments between the semeval data and the entity files
    base_dir = os.path.join(os.path.dirname(__file__), "data/semeval_train")

    # code adapted from pretrain.py with minor modifications
    # expected format: train -> [ar -> train.jsonl, de -> train.jsonl...]
    for folder_name in os.listdir(base_dir):
        print(folder_name) # so we can check that both are being processed in the same order
        folder_path = os.path.join(base_dir, folder_name)

        # check if the path is a language folder
        if os.path.isdir(folder_path):
            jsonl_file_path = os.path.join(folder_path, "train.jsonl")

            if os.path.isfile(jsonl_file_path):
                with open(jsonl_file_path, "r", encoding="utf-8") as jsonl_file:

                    lines = list(jsonl_file)
                    rows_per_df.append(len(lines))

                    data_target = [json.loads(line)["target"] for line in lines if "target" in json.loads(line)]
                    data_source = [json.loads(line)["source"] for line in lines if "source" in json.loads(line)]
                    target_locale = ["<" + jsonl_file_path.split("/")[-2] + ">" for line in lines]
                    

                    df = pd.DataFrame({"source": data_source, "target": data_target, "target_locale": target_locale})

                    sp = spm.SentencePieceProcessor(model_file="tokenizer/tokenizer_combined.model")
                    df["source"] = df["source"].apply(lambda text: sp.encode(text, out_type=str))
                    df["target"] = df["target"].apply(lambda text: sp.encode(text, out_type=str))

                    semeval_train.append(df)
    if just_get_lines:
        return rows_per_df
    else:
        return semeval_train
    
def get_semeval_val(just_get_lines = False): # knowing the lines will be used to check if the entities line up with the train dfs
    semeval_val = []
    rows_per_df = [] # once again, this will help us detect misallignments between the semeval data and the entity files
    base_dir = os.path.join(os.path.dirname(__file__), "data/semeval_val")
    
    if os.path.isdir(base_dir):
        for file_name in os.listdir(base_dir): 
            json_file_path = os.path.join(base_dir, file_name)
            with open(json_file_path, "r", encoding="utf-8") as jsonl_file:

                lines = list(jsonl_file)
                rows_per_df.append(len(lines))

                data_target = [json.loads(line)["targets"][0]["translation"] for line in lines if "targets" in json.loads(line)]
                data_source = [json.loads(line)["source"] for line in lines if "source" in json.loads(line)]
                target_locale = ["<" + json_file_path.split("_")[-1].split(".")[0] + ">" for line in lines]

                df = pd.DataFrame({"source": data_source, "target": data_target, "target_locale": target_locale})

                sp = spm.SentencePieceProcessor(model_file="tokenizer/tokenizer_combined.model")
                df["source"] = df["source"].apply(lambda text: sp.encode(text, out_type=str))
                df["target"] = df["target"].apply(lambda text: sp.encode(text, out_type=str))

                semeval_val.append(df)
    if just_get_lines:
        return rows_per_df
    else:
        return semeval_val


def get_entity_info(just_get_lines = False, train=True):
    entity_info = []
    num_rows = []
    if train:
        base_dir = os.path.join(os.path.dirname(__file__), "data/entity_info/train")
    else:
        base_dir = os.path.join(os.path.dirname(__file__), "data/entity_info/val")

    sp = spm.SentencePieceProcessor(model_file="tokenizer/tokenizer_combined.model")

    for df_name in os.listdir(base_dir):
        print(df_name) # so we can check that both are being processed in the same order
        csv_file_path = os.path.join(base_dir, df_name)

        if os.path.isfile(csv_file_path) and csv_file_path.endswith(".csv"):
            df = pd.read_csv(csv_file_path)
            num_rows.append(df.shape[0])


            if "source" in df.columns and "target" in df.columns:

                df["source"] = df["source"].apply(lambda text: sp.encode(str(text), out_type=str))
                df["target"] = df["target"].apply(lambda text: sp.encode(str(text), out_type=str))

                entity_info.append(df if isinstance(df, pd.DataFrame) else pd.DataFrame(df))
    if just_get_lines:
        return num_rows
    else:
        return entity_info

def make_dummy_entity_data(train=True):
    """
    Creates dummy CSV files with the specified number of rows for each file.

    """

    languages = ["ar", "de", "es", "fr", "it", "ja"]
    if train:
        base_dir = "data/entity_info/train"
    else:
        base_dir = "data/entity_info/val"


    lines = get_semeval_train(just_get_lines=True)  # the number of lines that should be in each df in a list

    if len(lines) != len(languages):
        raise ValueError("The number of rows must match the number of language files.")


    os.makedirs(base_dir, exist_ok=True)

    for lang, num_rows in zip(languages, lines):
        data = {
            "source": [f"dummy source entity {lang}"] * num_rows,
            "target": [f"dummy target entity {lang}"] * num_rows
        }
        df = pd.DataFrame(data)
        csv_path = os.path.join(base_dir, f"{lang}.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"Created file: {csv_path} with {num_rows} rows.")


# Chunking not done on training data b/c source and translation must line up
def collate_fn(batch):
    
    encoder_input = [item[0] for item in batch]
    decoder_input = [item[1] for item in batch]
    decoder_output = [item[2] for item in batch]
    mask = [item[3] for item in batch]
    print(batch.size)

    # set batch_first to True to make the batch size first dim
    padded_en_in = pad_sequence(encoder_input, batch_first=True, padding_value=semeval_dataset.vocab["<PAD>"])  # does not matter if semeval or pretrain, should be the same vocab
    padded_de_in = pad_sequence(decoder_input, batch_first=True, padding_value=semeval_dataset.vocab["<PAD>"])
    padded_de_out = pad_sequence(decoder_output, batch_first=True, padding_value=semeval_dataset.vocab["<PAD>"])
    padded_mask = pad_sequence(mask, batch_first=True, padding_value=semeval_dataset.vocab["<PAD>"])
    return padded_en_in, padded_de_in, padded_de_out, padded_mask


# Exact same logic as the collate_fn, except for entity info!
def entity_fn(batch):
    source = [item[0] for item in batch]
    target = [item[1] for item in batch]

    padded_source = pad_sequence(source, batch_first=True, padding_value=semeval_dataset.vocab["<PAD>"])
    padded_target = pad_sequence(target, batch_first=True, padding_value=semeval_dataset.vocab["<PAD>"])
    return padded_source, padded_target


# Encode and load pretrain data
make_dummy_entity_data()
make_dummy_entity_data(train=False)

print("running pretrain")
semeval_train = get_semeval_train()
pretrain_dataset = TranslationDataset()
pretrain_dataset.make_vocab(pretrain_list, semeval_train)
pretrain_dataset.encode_pretrain(pretrain_list)
pretrain_train, pretrain_val = train_test_split(pretrain_dataset, test_size=0.1, random_state=27)
# No padding in the pretrainig data
pretrain_train_loader = DataLoader(pretrain_train, batch_size=64, shuffle=True, collate_fn=collate_fn)
pretrain_val_loader = DataLoader(pretrain_val, batch_size=64, shuffle=True, collate_fn=collate_fn)
# print("🔎😮‍💨 analyzing pretrain dataset")
# pretrain_dataset.make_sure_everythings_alligned_properly()

# Encode and load train data
print("running train")
semeval_val = get_semeval_val()
entities_train = get_entity_info()
entities_val = get_entity_info(train=False)


semeval_dataset = TranslationDataset()
semeval_val_dataset = TranslationDataset()
semeval_dataset.make_vocab(pretrain_list, semeval_train, entities_train)

semeval_dataset.encode_semeval(semeval_train)
semeval_val_dataset.encode_semeval(semeval_val)
semeval_val_dataset.make_vocab(pretrain_list, semeval_train, entities_train)

semeval_train_loader = DataLoader(semeval_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
semeval_val_loader = DataLoader(semeval_val, batch_size=64, shuffle=True, collate_fn=collate_fn)


entity_train_info = DataLoader(entities_train, batch_size=64, shuffle=True, collate_fn=entity_fn)
entity_val_info = DataLoader(entities_val, batch_size=64, shuffle=True, collate_fn=entity_fn)

# # just some code for testing
# print("running test")
# semeval_train = get_semeval_train()
# entity_info = get_entity_info()
# entity_dataset = TranslationDataset()
# entity_dataset.make_vocab(pretrain_list, semeval_train, entity_info)

# entity_dataset.encode_semeval(semeval_train, entity_data=entity_info)

# print(entity_dataset.entity_ids)

# print("🔎😮‍💨 analyzing train dataset ")
# semeval_dataset.make_sure_everythings_alligned_properly()

