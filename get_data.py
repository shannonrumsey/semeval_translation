import pandas as pd
import os
import sys


def get_data(languages, exceptions, train=True):
    """
    Takes in a list of language file codes and creates a dataset that contains the source and target sentence, as well as the entity(ies) and translated entity(ies)
    Note: The exceptions are a list of languages that have val data that will only be used as training data and not val!
    """
    lang_data = []
    for lang in languages:

        # Convert lang to correct format for entity files
        entity_lang = lang.split("_")[0]

        # All train data
        if train and lang not in exceptions:

            sent_path = os.path.join(os.path.dirname(__file__), "data/semeval_train/") + lang + ".jsonl"
            entity_path = os.path.join(os.path.dirname(__file__), "data/entity_info/train/") + entity_lang + ".csv"
            sents = pd.read_json(sent_path, lines=True)[["id", "source", "target"]]

        # Val data to be included as "train data" or val data that excludes the exceptions
        elif (train and lang in exceptions) or (not train and lang not in exceptions):

            sent_path = os.path.join(os.path.dirname(__file__), "data/semeval_val/") + lang + ".jsonl"
            entity_path = os.path.join(os.path.dirname(__file__), "data/entity_info/val/") + entity_lang + ".csv"

            sents = pd.read_json(sent_path, lines=True)
            sents["targets"] = sents["targets"].apply(lambda x: x[0]["translation"] if isinstance(x, list) else x)

            sents = sents[["id", "source", "targets"]].rename(columns={"targets": "target"})
  
        entities = pd.read_csv(entity_path)

        # Rename entity columns since they are the same as the sents dataframe
        entities = entities.rename(columns={"source": "entity_src", "target": "entity_trg"})

        entities["lang"] = lang

        lang_data.append(pd.concat([sents, entities], axis=1))
        
    return pd.concat(lang_data, ignore_index=True)

def get_test_data(languages):
    lang_data = []
    for lang in languages:

        # Convert lang to correct format for entity files
        entity_lang = lang.split("_")[0]

        sent_path = os.path.join(os.path.dirname(__file__), "data/semeval_test/") + lang + ".jsonl"
        entity_path = os.path.join(os.path.dirname(__file__), "data/entity_info/test/") + entity_lang + ".csv"

        sents = pd.read_json(sent_path, lines=True)

        entities = pd.read_csv(entity_path)

        # Rename entity columns since they are the same as the sents dataframe
        entities = entities.rename(columns={"source": "entity_src", "target": "entity_trg"})

        entities["lang"] = lang
        
        lang_data.append(pd.concat([sents["id"], sents["source"], entities], axis=1))

    return pd.concat(lang_data, ignore_index=True)

