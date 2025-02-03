import json
import pandas
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

path = lambda x: os.path.join(os.path.dirname(__file__),
                              x)  # relative path lambda abuse, call lambda abuse hotline for more details

# define dataset
print("defining the dataset")


class TranslationDataset(Dataset):
    """
    Loads the data from the CSV files generated in pretrain.py
    """

    def __init__(self, vocab=None):
        if vocab != None:
            self.vocab = vocab
        else:
            self.vocab = {"<PAD>": 0, "<unk>": 1, "[ent_info]": 2,
                          "->": 3}  # special symbols which may not be in the data but need to be included

        self.inverse_vocab = None
        self.corpus_encoder_ids = []
        self.corpus_decoder_ids = []
        self.corpus_target_ids = []
        self.corpus_y_mask = []  # In the case of pretrain, this will be a dummy variable (all 1s, indicating no padding)
        self.entity_ids = None

    def make_vocab(self, pretrain_data, train_data, entity_data=None, val_data=None):
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
                else:  # in some of the dfs, the lists got saved as strings
                    try:
                        row = ast.literal_eval(row)
                        if isinstance(row, list):
                            pass
                            # print("Successfully converted to list:", row)
                        else:
                            print(f"row is not a list, it is of type {type(row)}: {row}")
                    except (ValueError, SyntaxError):
                        print("Error: The string could not be interpreted as a list.")
                for token in row:

                    if token not in self.vocab:
                        self.vocab[token] = len(self.vocab)

            for row in df[
                "decoder_input"]:  # most of these will probably already be in the input, but the masked ones might not be
                if isinstance(row, list):
                    pass
                else:  # in some of the dfs, the lists got saved as strings
                    try:
                        row = ast.literal_eval(row)
                        if isinstance(row, list):
                            pass
                            # print("Successfully converted to list:", row)
                        else:
                            print(f"row is not a list, it is of type {type(row)}: {row}")
                    except (ValueError, SyntaxError):
                        print("Error: The string could not be interpreted as a list.")
                for token in row:

                    if token not in self.vocab:
                        self.vocab[token] = len(self.vocab)

            # we shouldnt need to do this on the decoder outputs because they will be the same

        # add tokens in train data to vocab as well
        for key in train_data:
            df = train_data[key]
            print(df.head(2))
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
                            # print("Successfully converted to list:", row)
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
        if val_data:
            for key in val_data:
                print("VAL DATA")
                df = val_data[key]
                print(df.head(2))
                if isinstance(df, pd.DataFrame):
                    print("val is a df")
                else:
                    print("val is NOT a df")

                for row in df["source"]:
                    if isinstance(row, list):
                        pass
                    else:  # in some of the dfs, the lists got saved as strings
                        try:
                            row = ast.literal_eval(row)
                            if isinstance(row, list):
                                pass
                                # print("Successfully converted to list:", row)
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
            for key in entity_data:
                df = entity_data[key]
                if isinstance(df, pd.DataFrame):
                    print("entity_data is a df")
                else:
                    print(df)
                    print(type(df))

                    print("entity_data is NOT a df")

                for row in df["target"]:

                    for entity in row:  # we are assuming source contains lists of entity bpes

                        for token in entity:

                            if token not in self.vocab:
                                self.vocab[token] = len(self.vocab)

                for row in df["source"]:  # zero is source
                    for entity in row:
                        for token in entity:
                            if token not in self.vocab:
                                self.vocab[token] = len(self.vocab)

        self.inverse_vocab = {index: token for token, index in self.vocab.items()}

        #self.save_vocabs_to_file('first') #uncomment to save vocabs to files in ./vocab

    def save_vocabs_to_file(self, prefix:str):
        vocab_dir = path('./vocab')
        try:
            with open(f'{vocab_dir}/{prefix}_vocab.json', 'w') as f:
                f.write(json.dumps(self.vocab, indent=3))
            with open(f'{vocab_dir}/{prefix}_inv_vocab.json', 'w') as f:
                f.write(json.dumps(self.inverse_vocab, indent=3))
        
            return True
        except:
            return False


    def load_vocab(self, vocab):  # this method will be used for val amd test sets to ensure they have the same vocab
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
                self.corpus_encoder_ids.append(torch.tensor(encoder_ids))

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
                self.corpus_decoder_ids.append(torch.tensor(decoder_ids))

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

    def encode_semeval(self, data, entity_data=None, train=True):  # for training, val, amd test semeval data
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
        def lazy_debug(idx):
            try:
                print(self.corpus_encoder_ids[idx].shape)
                string = ""
                for x in self.corpus_encoder_ids[idx]:
                    string += self.inverse_vocab[x.item()]
                print(string)
            except:
                print(f'error accessing index: {idx}')
        
        if train:
            lang_processing_order = ["ar", "de", "es", "fr", "it", "ja", "ko", "th", "tr", "zh"]
        else:  # for val, we only use the languages we have val on
            lang_processing_order = ["ar", "de", "es", "fr", "it", "ja"]
        if self.vocab is None:
            raise ValueError("🚩No vocab found 🚩. Please build vocab using 'make_vocab()' and try again.")
        print("printing self.vocab")
        # print(self.vocab)
        for key in lang_processing_order:
            print("key being processed: ", key)
            df = data[key]
            source = df["source"]
            target = df["target"]
            lang = df["target_locale"]

            for src, trg, l in zip(source, target, lang):
                encoder_input = [l] + src + ["</s>"]
                encoder_ids = [self.vocab.get(token, self.vocab['<unk>']) for token in encoder_input]

                decoder_input = [l] + trg + ["</s>"]
                decoder_ids = [self.vocab.get(token, self.vocab['<unk>']) for token in decoder_input]

                target = decoder_input[1:] + [
                    l]  # Add random token to end to ensure deocder input length = target length
                target_ids = [self.vocab.get(token, self.vocab['<unk>']) for token in target]

                self.corpus_encoder_ids.append(torch.tensor(encoder_ids))
                self.corpus_decoder_ids.append(torch.tensor(decoder_ids))
                self.corpus_target_ids.append(torch.tensor(target_ids))
                dummy_mask = torch.ones(len(target_ids))
                self.corpus_y_mask.append(dummy_mask)

        if entity_data:
            self.entity_ids = []

            for key in lang_processing_order:
                print("key being processed: ", key)
                df = entity_data[key]  # The koreans are pissing this off (🇰🇵)
                source = df["source"]  # source is going to be like ["Be yon ce", "Dens tiny Child"]
                target = df["target"]  # target is going to be like ["Be yon ce", "Hi jo de Des tino"]

                for s, t in zip(source,
                                target):  # Note: this will only be the entity data, (source and target translations for just the entity)

                    full_sentence = []
                    if len(s) != len(t):
                        print("\nthe error with lens occured🧘‍♀️😌🧘‍♀️😌")
                        print(t)
                        print(s)
                        print("\n")
                    for entity_index in range(len(s)):
                        s_val = s[entity_index]
                        t_val = t[entity_index]

                        if "НИЧЕГО" in s_val or "НИЧЕГО" in t_val:
            
                            entity_sentence = ""
                        else:
                            s_tokens = [self.vocab.get(token, self.vocab['<unk>']) for token in s[entity_index]]
                            t_tokens = [self.vocab.get(token, self.vocab['<unk>']) for token in t[entity_index]]

                            entity_sentence = [self.vocab["[ent_info]"]] + s_tokens + [self.vocab[
                                                                                       "->"]] + t_tokens  # creates individual encodings and translations for each entity

                        # example ["[ent_info]", "Be", "yon", "ce" "->", "Be", "yon", "ce" ]
                        # example ["[ent_info]", "Dens", "tiny", "Child" "->", "Hi", "jo", "de", "Des", "tino"]

                        full_sentence += entity_sentence

                    # concant the lists for all the entities in the sentence together into a single sequence
                    # example["[ent_info]", "Be", "yon", "ce" "->", "Be", "yon", "ce", "[ent_info]", "Dens", "tiny", "Child" "->", "Hi", "jo", "de", "Des", "tino"]
                    self.entity_ids.append(torch.tensor(full_sentence))

        # concatinates entity info to the encoder if entity info is present
        if self.entity_ids is not None:
            idx = len(self.entity_ids) - 1 #get max - 1 so we never trigger an error here even in val
            print("we are merging entities with inputs")
            print("\n\n❤️😍😘BABY DOLL START PAYING ATTENTION ❤️😍😘❤️😍😘❤️😍😘\n")
            print("🥭🥭ORIGINAL SHAPE OF FIRST ENCODER ID")

            print(self.corpus_encoder_ids[600].shape)
            string = ""
            for x in self.corpus_encoder_ids[600]:
                string += self.inverse_vocab[x.item()]
            print(string)

            print("🥭🥭ORIGINAL SHAPE OF FIRST ENTITY ID")
            print(self.entity_ids[600].shape)
            string = ""
            for x in self.entity_ids[600]:
                string += self.inverse_vocab[x.item()]
            print(string)



            print("🥭🥭ORIGINAL SHAPE OF FIRST ENTITY ID")


            self.corpus_encoder_ids = [
                torch.cat((c, e), dim=-1) for c, e in zip(self.corpus_encoder_ids, self.entity_ids)
            ]
            print("🥭🥭AFTER SHAPE OF FIRST ENCODER ID")

            print(self.corpus_encoder_ids[600].shape)
            string = ""
            for x in self.corpus_encoder_ids[600]:
                string += self.inverse_vocab[x.item()]
            print(string)

            print("🥭🥭AFTER SHAPE OF FIRST DECODER ID")
            print(self.corpus_decoder_ids[600].shape)
            string = ""
            for x in self.corpus_decoder_ids[600]:
                string += self.inverse_vocab[x.item()]
            print(string)

            print("🥭🥭AFTER SHAPE OF FIRST TARGET ID")
            print(self.corpus_target_ids[600].shape)
            string = ""
            for x in self.corpus_target_ids[600]:
                string += self.inverse_vocab[x.item()]
            print(string)


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

    def make_sure_everythings_alligned_properly(self, train=False):
        print("making sure everything looks good in the dataset")
        random_index = random.randint(0, len(self.corpus_encoder_ids) - 1)  # to get a random sample from the data
        print("fetching data at random index: ", random_index)
        encoder = self.corpus_encoder_ids[random_index]
        decoder = self.corpus_decoder_ids[random_index]
        target = self.corpus_target_ids[random_index]
        mask = self.corpus_y_mask[random_index]
        if train is False:
            print("\n====== CHECKING LENGTHS =====")
            print("encoder_ids: ", len(encoder), "decoder_ids: ", len(decoder), "target_ids: ", len(target),
                  "mask_ids: ", len(mask))
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

        # print("\n======= numerical values ======")
        # print("\nencoder_ids: ", encoder, "\ndecoder_ids: ", decoder, "\ntarget_ids: ", target, "\nmask_ids: ", mask)

        real_encoder = [self.inverse_vocab.get(token.item(), "<unk>") for token in encoder]
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
            return self.corpus_encoder_ids[idx], self.corpus_decoder_ids[idx], self.corpus_target_ids[idx], \
            self.corpus_y_mask[idx]


pretrain_list = []
folder_path = os.path.abspath("data/processed_pretrain")

for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        pretrain_list.append(df)


def get_semeval_train(
        just_get_lines=False):  # knowing the lines will be used to check if the entities line up with the train dfs
    semeval_train = {}
    rows_per_df = []  # once again, this will help us detect misallignments between the semeval data and the entity files
    base_dir = os.path.join(os.path.dirname(__file__), "data/semeval_train")

    # code adapted from pretrain.py with minor modifications
    # expected format: train -> [ar -> train.jsonl, de -> train.jsonl...]
    for file in os.listdir(base_dir):
        jsonl_file_path = os.path.join(base_dir, file)

        # check if the path is a language folder
        lang_name = file.split("_")[0]  # Tucker Carlson levels of hackyness
        print("train lang name: ", lang_name)
        try:
            with open(jsonl_file_path, "r", encoding="utf-8") as jsonl_file:

                lines = list(jsonl_file)
                rows_per_df.append(len(lines))

                data_target = [json.loads(line)["target"] for line in lines if "target" in json.loads(line)]
                data_source = [json.loads(line)["source"] for line in lines if "source" in json.loads(line)]
                target_locale = [f"<{lang_name}>" for line in lines]

                df = pd.DataFrame({"source": data_source, "target": data_target, "target_locale": target_locale})

                sp = spm.SentencePieceProcessor(model_file="tokenizer/tokenizer_combined.model")
                df["source"] = df["source"].apply(lambda text: sp.encode(text, out_type=str))
                df["target"] = df["target"].apply(lambda text: sp.encode(text, out_type=str))

                semeval_train[lang_name] = df

        except:
            print(
                f"🙈🙈🙈 WHOOPSIE DASIE 🙈🙈🙈\nFile: {file} is cursed. Consider casting a spell to counter\nYou are fine if this is the .DS_STORE")

    # Get val datasets for the missing languages
    val_dir = os.path.join(os.path.dirname(__file__), "data/semeval_val")
    exceptions = ["ko_KR", "th_TH", "tr_TR", "zh_TW"]
    if os.path.isdir(val_dir):
        for file_name in os.listdir(val_dir):
            base_name = os.path.splitext(file_name)[0]

            if base_name in exceptions:
                lang_name = base_name.split("_")[0]
                print("val used as train lang_name: ", lang_name)

                json_file_path = os.path.join(val_dir, file_name)
                with open(json_file_path, "r", encoding="utf-8") as jsonl_file:
                    lines = list(jsonl_file)
                    rows_per_df.append(len(lines))

                    data_target = [json.loads(line)["targets"][0]["translation"] for line in lines if
                                   "targets" in json.loads(line)]
                    data_source = [json.loads(line)["source"] for line in lines if "source" in json.loads(line)]
                    target_locale = ["<" + base_name.split("_")[0] + ">" for line in lines]

                    df = pd.DataFrame({"source": data_source, "target": data_target, "target_locale": target_locale})
                    sp = spm.SentencePieceProcessor(model_file="tokenizer/tokenizer_combined.model")
                    df["source"] = df["source"].apply(lambda text: sp.encode(text, out_type=str))
                    df["target"] = df["target"].apply(lambda text: sp.encode(text, out_type=str))

                    semeval_train[lang_name] = df

    if just_get_lines:
        return rows_per_df
    else:
        return semeval_train


def get_semeval_val(
        just_get_lines=False):  # knowing the lines will be used to check if the entities line up with the train dfs
    semeval_val = {}
    rows_per_df = []  # once again, this will help us detect misallignments between the semeval data and the entity files
    base_dir = os.path.join(os.path.dirname(__file__), "data/semeval_val")

    exceptions = ["ko_KR", "th_TH", "tr_TR", "zh_TW"]
    if os.path.isdir(base_dir):
        for file_name in os.listdir(base_dir):
            base_name = os.path.splitext(file_name)[0]

            if base_name not in exceptions:
                lang_name = base_name.split("_")[0]
                print("lang_name in val: ", lang_name)
                json_file_path = os.path.join(base_dir, file_name)
                with open(json_file_path, "r", encoding="utf-8") as jsonl_file:
                    lines = list(jsonl_file)
                    rows_per_df.append(len(lines))

                    data_target = [json.loads(line)["targets"][0]["translation"] for line in lines if
                                   "targets" in json.loads(line)]
                    data_source = [json.loads(line)["source"] for line in lines if "source" in json.loads(line)]
                    target_locale = ["<" + base_name.split("_")[0] + ">" for line in lines]

                    df = pd.DataFrame({"source": data_source, "target": data_target, "target_locale": target_locale})

                    sp = spm.SentencePieceProcessor(model_file="tokenizer/tokenizer_combined.model")
                    df["source"] = df["source"].apply(lambda text: sp.encode(text, out_type=str))
                    df["target"] = df["target"].apply(lambda text: sp.encode(text, out_type=str))

                    semeval_val[lang_name] = df
    if just_get_lines:
        return rows_per_df
    else:
        return semeval_val


def get_entity_info(just_get_lines=False, train=True):
    get_lang = lambda x: x.split('.')[0]  # cheeky ass mf lambda (report lambda abuse)
    entity_info = {}
    num_rows = []

    if train:
        base_dir = path("data/entity_info/train")
    else:
        base_dir = path("data/entity_info/val")

    sp = spm.SentencePieceProcessor(model_file="tokenizer/tokenizer_combined.model")

    def add_lang_to_ent_info(lang, file):  # subprocess to add our language to the ent info for DRY
        df = pd.read_csv(file)  # load csv
        df["source"] = df["source"].fillna("НИЧЕГО")
        df["target"] = df["target"].fillna("НИЧЕГО")
        num_rows.append(df.shape[
                            0])  # idk this was in the old loop so i kept it but seems not needed unless "just_get_lines = true"
        df["source"] = df["source"].apply(
            lambda text: [sp.encode(entity, out_type=str) for entity in str(text).split("*|*")])
        df["target"] = df["target"].apply(
            lambda text: [sp.encode(entity, out_type=str) for entity in str(text).split("*|*")])

        entity_info[lang] = df

    for df_name in os.listdir(base_dir):  # main loop
        try:
            print("entity df name: ", df_name)
            csv_file_path = os.path.join(base_dir, df_name)
            entity_lang = get_lang(df_name)

            add_lang_to_ent_info(entity_lang, csv_file_path)
        except:
            print(f'👺 YOU FUCKED UP\nBURN IN HELL🔥🔥🔥\n' if df_name.find(
                '.DS') == -1 else f'Calmate\nNo real problem')  # if

    if train:  # if training we want these validation entity files
        val_path = path('data/entity_info/val')
        needed = set(['ko.csv', 'th.csv', 'tr.csv', 'zh.csv'])
        for file in os.listdir(val_path):  # iterate over validation entity files
            if file in needed:  # grab needed ones
                lang = get_lang(file)
                add_lang_to_ent_info(lang, f'{val_path}/{file}')

    if just_get_lines:
        return num_rows
    else:
        return entity_info


def make_dummy_entity_data(train=True):
    """
    Creates dummy CSV files with the specified number of rows for each file.

    """
    print("NOOO FUCK YOU! STOP TRYING TO REPLACE JACKS CODE FUCKKKKKKK YOUUUUUUU")
    '''
    if train:
        base_dir = "data/entity_info/train"
        lines = get_semeval_train(just_get_lines=True)  # the number of lines that should be in each df in a list
        languages = ["ar", "de", "es", "fr", "it", "ja", "ko", "th", "tr", "zh"]
    else:
        base_dir = "data/entity_info/val"
        lines = get_semeval_val(just_get_lines=True)  # the number of lines that should be in each df in a list
        print("lines for val: ", lines)
        languages = ["ar", "de", "es", "fr", "it", "ja"]





    os.makedirs(base_dir, exist_ok=True)

    for lang, num_rows in zip(languages, lines):
        data = {
            "source": [f"Moscow {lang}*|*North Dakota*|*New York"] * num_rows,
            "target": [f"Moscú {lang}*|*Dakota del Norte*|*Nueva York"] * num_rows
        }
        df = pd.DataFrame(data)
        csv_path = os.path.join(base_dir, f"{lang}.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"Created file: {csv_path} with {num_rows} rows.")'''


# Chunking not done on training data b/c source and translation must line up
def collate_fn(batch):
    entities = None

    if all(len(item) == 5 for item in batch):
        entities = [item[4] for item in batch]
    encoder_input = [item[0] for item in batch]
    decoder_input = [item[1] for item in batch]
    decoder_output = [item[2] for item in batch]
    mask = [item[3] for item in batch]

    # set batch_first to True to make the batch size first dim
    padded_en_in = pad_sequence(encoder_input, batch_first=True, padding_value=semeval_train_dataset.vocab[
        "<PAD>"])  # does not matter if semeval or pretrain, should be the same vocab
    padded_de_in = pad_sequence(decoder_input, batch_first=True, padding_value=semeval_train_dataset.vocab["<PAD>"])
    padded_de_out = pad_sequence(decoder_output, batch_first=True, padding_value=semeval_train_dataset.vocab["<PAD>"])
    padded_mask = pad_sequence(mask, batch_first=True, padding_value=semeval_train_dataset.vocab["<PAD>"])
    # print("🌻🌻 testing random encoder id inside the collate function to make sure its behaving:")
    # if len(padded_en_in) >  5:
    #     for en in padded_en_in[:5]:
    #         test_string = ""
    #         for item in en:
    #             test_string += semeval_train_dataset.inverse_vocab[item.item()] + " "
    #         print(test_string)

    if entities is not None:
        padded_entities = pad_sequence(entities, batch_first=True, padding_value=semeval_train_dataset.vocab["<PAD>"])
        return padded_en_in, padded_de_in, padded_de_out, padded_mask, padded_entities

    else:
        return padded_en_in, padded_de_in, padded_de_out, padded_mask


# Encode and load pretrain data
semeval_train = get_semeval_train()
semeval_val = get_semeval_val()
entities_train = get_entity_info()
entities_val = get_entity_info(train=False)


print("running pretrain")
pretrain_dataset = TranslationDataset()
pretrain_dataset.make_vocab(pretrain_list, semeval_train, entities_train, semeval_val)
pretrain_dataset.encode_pretrain(pretrain_list)
pretrain_train, pretrain_val = train_test_split(pretrain_dataset, test_size=0.1, random_state=27)
# No padding in the pretrainig data
pretrain_train_loader = DataLoader(pretrain_train, batch_size=64, shuffle=True, collate_fn=collate_fn)
pretrain_val_loader = DataLoader(pretrain_val, batch_size=64, shuffle=True, collate_fn=collate_fn)
print("🔎😮‍💨 analyzing pretrain dataset")
pretrain_dataset.make_sure_everythings_alligned_properly()

# Encode and load train data
print("running train")

semeval_train_dataset = TranslationDataset()
semeval_val_dataset = TranslationDataset()

semeval_train_dataset.load_vocab(pretrain_dataset.vocab)
semeval_val_dataset.load_vocab(pretrain_dataset.vocab)

semeval_train_dataset.encode_semeval(semeval_train, entity_data=entities_train)  # Problem child ATM


entities_val = get_entity_info(train=False)
print("printing entity head")
for key in entities_val:
    print("key in entity val")
    print(key)
semeval_val_dataset.encode_semeval(semeval_val, entity_data =entities_val,  train=False)  # NOTE: need to add val entities


semeval_train_loader = DataLoader(semeval_train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
semeval_val_loader = DataLoader(semeval_val_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

# print("🟥🟥testing a random encoder id:")
# test_string = ""
# for item in semeval_train_dataset.corpus_encoder_ids[30]:
#     test_string += semeval_train_dataset.inverse_vocab[item.item()] + " "
# print(test_string)
# print("decoder len: ", len(semeval_train_dataset.corpus_decoder_ids))


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