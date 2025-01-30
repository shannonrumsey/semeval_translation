import flair
from flair.data import Sentence
from flair.nn import Classifier
import torch
import pandas as pd
import time
import os
from knowledge import KnowledgeBase
from tqdm import tqdm

seed = 69
torch.manual_seed(seed)

lang_map = { #languages mapped to their file names 
    'ar': 'ar_AE',
    'de': 'de_DE',
    'es': 'es_ES',
    'fr': 'fr_FR',
    'it': 'it_IT',
    'ja': 'ja_JP',
    'ko': 'ko_KR',
    'th': 'th_TH',
    'tr': 'tr_TR',
    'zh': 'zh_TW'
}


# setting the device
def get_torch_device():
    if torch.cuda.is_available():
        #print("Using CUDA")
        device = torch.device("cuda:0")
        flair.device = 'cuda:0'

    elif torch.backends.mps.is_available():
       #print("Using MPS")
        device = torch.device("mps")
        flair.device = 'mps:0'

    else:
        #print("Using CPU")
        device = torch.device("cpu")
        flair.device = 'cpu'
    return device

# load the NER tagger
tagger = Classifier.load('ner-large').to(device=get_torch_device())


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

def ner_predictor(language: str, output_file: str, verbose: bool = False, directory='semeval_val'):
    """
    Predicts the entities in the given JSON file and saves the predictions to the output file.
    Args:
        language (str): The language of the data to be predicted.
        output_file (str): The path to the output file where the predictions will be saved.
        verbose (bool): Whether to print the predictions to the console.
        directory (str): Set to the directory we want to pull data from: train='semeval_train', val='semeval_val', test='semeval_test')
    """
    start_time = time.time()
    path = os.path.join(os.path.dirname(__file__), 'data', directory, f'{lang_map[language]}.jsonl')
    output_path = os.path.join(os.path.dirname(__file__), 'data', 'entity_info', directory, output_file)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    reader = pd.read_json(path, lines=True)
    predictions = []
    translated_entities = []

    device = get_torch_device()

    
    # create KB class
    kb = KnowledgeBase()

    for _, row in tqdm(reader.iterrows(), desc=f'Building entity data for {language}'):
        # make a sentence
        sentence = Sentence(row['source'])
        sentence.to(device=device)
        with torch.no_grad():
            tagger.predict(sentence)
    
        entities = []
        entity_translation = []
        for entity in sentence.get_spans('ner'):
            found = kb.get(entity.text, language)
            if found:
                entities.append(entity.text)
                entity_translation.append(str(found))
        
        if verbose:
            print(f'ENTS: {entities}\tTRANS: {entity_translation}')
        predictions.append('*|*'.join(entities))
        translated_entities.append('*|*'.join(entity_translation))

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

def build_files():
    """Iterate over our datasets and build out entity predictions on them"""
    for di in ['semeval_train', 'semeval_val', 'semeval_test']:
        for lang, lang_name in lang_map.items():
            path = os.path.join(os.path.dirname(__file__), 'data', 'entity_info', di.split('_')[1], f'{lang}.csv')
            print(path)
            try:
                if os.path.exists(path):
                    print('exists')
                    continue
                ner_predictor(lang, f'{lang_name}.csv', directory=di)
            except Exception as e:
                print(f'Encountered: {e}\nWhile using {lang}, {lang_name} in DIRECTORY: {di}')

#build_files() #uncomment this line to rebuild entity prediction files
""" 
example usage of knowledge base

kb = KnowledgeBase()
print(kb.get_entity_from_id("Q8877")) 
"""