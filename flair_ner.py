import flair
from flair.data import Sentence
from flair.nn import Classifier
import torch
import pandas as pd
import time
import os

seed = 69
torch.manual_seed(seed)

# setting the device
def get_torch_device():
    if torch.cuda.is_available():
        print("Using CUDA")
        device = torch.device("cuda:0")
        flair.device = 'cuda:0'

    elif torch.backends.mps.is_available():
        print("Using MPS")
        device = torch.device("mps")
        flair.device = 'mps:0'

    else:
        print("Using CPU")
        device = torch.device("cpu")
        flair.device = 'cpu'
    return device

def ner_predictor(language: str, json_file: str, output_file: str, verbose: bool = False):
    """
    Predicts the entities in the given JSON file and saves the predictions to the output file.
    Args:
        language (str): The language of the data to be predicted.
        json_file (str): The path to the JSON file containing the data to be predicted.
        output_file (str): The path to the output file where the predictions will be saved.
        verbose (bool): Whether to print the predictions to the console.
    """
    start_time = time.time()
    path = f'data/semeval_train/{language}'
    os.chdir(path)
    reader = pd.read_json(json_file, lines=True)

    predictions = []

    device = get_torch_device()

    # load the NER tagger
    tagger = Classifier.load('ner-large').to(device=device)

    for _, row in reader.iterrows():
        
        # make a sentence
        sentence = Sentence(row['source'])
        sentence.to(device=device)
        with torch.no_grad():
            tagger.predict(sentence)
    
        entities = ""
        for entity in sentence.get_spans('ner'):
            entities += entity.text + "*|*"
        
        if verbose:
            print(entities[:-3])
        predictions.append(entities[:-3])

    # Create DataFrame with the predictions
    writer = pd.DataFrame({'source': predictions, 'target': predictions}) # TODO: change this to the actual target entities

    writer.index.name = 'id'

    writer.to_csv(output_file)

    end_time = time.time()
    if verbose:
        print(f"Time taken: {end_time - start_time} seconds")

# example usage
# ner_predictor('ar', 'train.jsonl', 'predictions_new.csv', verbose=True)