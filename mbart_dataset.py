import pandas as pd
import os
from get_data import get_data, get_test_data

languages = ["ar_AE", "de_DE", "es_ES", "fr_FR", "it_IT", "ja_JP", "ko_KR", "th_TH", "tr_TR", "zh_TW"]
# languages that don't have test data, use val data
exceptions = ["ko_KR", "th_TH", "tr_TR", "zh_TW"]

# Create datasets
val_data = get_data(languages=languages, exceptions=exceptions, train=False)
train_data = get_data(languages=languages, exceptions=exceptions, train=True)
assert val_data.iloc[-1]["target"] != train_data.iloc[-1]["target"], "Exceptions are being incorrectly handled!"
test_data = get_test_data(languages=languages)


# Creating mBART inputs
# current format: America is very beautiful<Amerika>, America and Turkey are very beautiful<Amerika,Türkiye>
def process_row(row):
    if pd.isna(row["entity_trg"]):
        return str(row["source"])
    
    else:
        split_entities = str(row["entity_trg"]).split("*|*")
        split_entities = [entity for entity in split_entities if pd.notna(entity) and entity.lower() != "nan"]

        return str(row["source"]) + "|" + ",".join(split_entities)


def mbart_data(dataset, path, test=False):
    """
    Must take in dataset that is outputted from get_data or get_test_data function.
    Saves the datasets to file at provided path.
    Use test=True when dataset is test data to exclude targets.
    """
    if test:
        df = pd.DataFrame({"id": dataset["id"], "lang": dataset["lang"]})
    else:
        df = pd.DataFrame({"id": dataset["id"], "lang": dataset["lang"], "target": dataset["target"]})
    
    df["source_entity"] = dataset.apply(lambda row: process_row(row), axis=1)

    df.to_csv(path)

base = os.path.join(os.path.dirname(__file__), "mbart_data/")
mbart_data(dataset=train_data, path=base+"train.csv", test=False)
mbart_data(dataset=test_data, path=base+"test.csv", test=True)
mbart_data(dataset=val_data, path=base+"val.csv", test=False)

