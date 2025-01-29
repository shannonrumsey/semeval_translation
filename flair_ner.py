import flair
from flair.data import Sentence
from flair.nn import Classifier
import torch
import pandas as pd
import time
import os
from knowledge import KnowledgeBase

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

def gather_true_entities(language: str, output_file: str):
    """
    Gathers the true entities from the given JSON file and saves them to the output file.
    Args:
        language (str): The language of the data to be predicted.
        output_file (str): The path to the output file where the predictions will be saved.
    Returns:
        true_entities (list): A list of the true entities.
    """
    path = os.path.join(os.path.dirname(__file__), 'data', 'semeval_train', language)
    
    json_path = os.path.join(path, "train.jsonl")
    output_path = os.path.join(path, output_file)
    reader = pd.read_json(json_path, lines=True)
    kb = KnowledgeBase()
    true_entities = []
    for _, row in reader.iterrows():
        entities = ""
        for entity in row["entities"]:
            found = kb.get_entity_from_id(entity)
            if found:
                entities += found + "*|*"
        true_entities.append(entities[:-3])
    writer = pd.DataFrame({'target': true_entities})

    writer.to_csv(output_path, index = False)

def ner_predictor(language: str, output_file: str, verbose: bool = False):
    """
    Predicts the entities in the given JSON file and saves the predictions to the output file.
    Args:
        language (str): The language of the data to be predicted.
        output_file (str): The path to the output file where the predictions will be saved.
        verbose (bool): Whether to print the predictions to the console.
    """
    start_time = time.time()
    path = os.path.join(os.path.dirname(__file__), 'data', 'semeval_train', language)
    output_path = os.path.join(os.path.dirname(__file__), 'data', 'entity_info', 'train', output_file)
        
    json_path = os.path.join(path, "train.jsonl")
    reader = pd.read_json(json_path, lines=True)

    predictions = []
    translated_entities = []

    device = get_torch_device()

    # load the NER tagger
    tagger = Classifier.load('ner-large').to(device=device)

    # create KB class
    kb = KnowledgeBase()

    for _, row in reader.iterrows():

        # make a sentence
        sentence = Sentence(row['source'])
        sentence.to(device=device)
        with torch.no_grad():
            tagger.predict(sentence)
    
        entities = ""
        entity_translation = ""
        for entity in sentence.get_spans('ner'):
            found = kb.get(entity.text, language)
            if found:
                entities += entity.text + "*|*"
                entity_translation += str(found) + "*|*"
        
        if verbose:
            print(entities[:-3] + ", " + entity_translation[:-3])
        predictions.append(entities[:-3])
        translated_entities.append(entity_translation[:-3])

    # Create DataFrame with the predictions
    writer = pd.DataFrame({'source': predictions, 'target': translated_entities})

    writer.to_csv(output_path, index = False)

    end_time = time.time()
    if verbose:
        print(f"Time taken: {end_time - start_time} seconds")

def ner_predictor_validation(language: str, output_file: str, type: str, verbose: bool = False):
    """
    Predicts the entities in the given JSON file and saves the predictions to the output file.
    Args:
        language (str): The language of the data to be predicted.
        output_file (str): The path to the output file where the predictions will be saved.
        type (str): How the predicted data will be used. Can be 'train' or 'val'.
        verbose (bool): Whether to print the predictions to the console.
    """
    start_time = time.time()
    if language == "ar":
        upper_language = "AE"
    elif language == "ja":
        upper_language = "JP"
    elif language == "ko":
        upper_language = "KR"
    elif language == "zh":
        upper_language = "TW"
    else:
        upper_language = language.upper()
    json_path = os.path.join(os.path.dirname(__file__), 'data', 'semeval_val', f'{language}_{upper_language}.jsonl')
    output_path = os.path.join(os.path.dirname(__file__), 'data', 'entity_info', type, output_file)

    reader = pd.read_json(json_path, lines=True)

    predictions = []
    translated_entities = []

    device = get_torch_device()

    # load the NER tagger
    tagger = Classifier.load('ner-large').to(device=device)
    
    # create KB class
    kb = KnowledgeBase()

    for _, row in reader.iterrows():

        # make a sentence
        sentence = Sentence(row['source'])
        sentence.to(device=device)
        with torch.no_grad():
            tagger.predict(sentence)
    
        entities = ""
        entity_translation = ""
        for entity in sentence.get_spans('ner'):
            found = kb.get(entity.text, language)
            if found:
                entities += entity.text + "*|*"
                entity_translation += str(found) + "*|*"
        
        if verbose:
            print(entities[:-3] + ", " + entity_translation[:-3])
        predictions.append(entities[:-3])
        translated_entities.append(entity_translation[:-3])

    # Create DataFrame with the predictions
    writer = pd.DataFrame({'source': predictions, 'target': translated_entities})

    writer.to_csv(output_path, index = False)

    end_time = time.time()
    if verbose:
        print(f"Time taken: {end_time - start_time} seconds")


# example usage
# gather_true_entities('de', 'true_entities.csv')
# ner_predictor('de', 'de.csv', verbose=True)
# ner_predictor_validation('de', 'de_val.csv', 'train', verbose=True)

""" 
example usage of knowledge base

kb = KnowledgeBase()
print(kb.get_entity_from_id("Q8877")) 
"""