import os
import pandas as pd
import sys
from tqdm import tqdm


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import semeval_langs, HF_MBART_CONFIG

rel_path = lambda x: os.path.join(os.path.dirname(__file__), x) #i abuse this lambda call so  much i should just make it a function 

def load_test_dataframe(lang='ar'):
    """
    Load a test file in based on the language
    """
    file_name = f'../data/semeval_test/{semeval_langs[lang]}.jsonl'
    file_path = rel_path(file_name)
    json_df = pd.read_json(file_path, lines=True)

    return json_df

def load_ent_test_info(lang:str):
    """
    Load ner predictions for test info based on a language
    """
    p = os.path.join(os.path.dirname(__file__), f'../data/entity_info/test/{lang}.csv')
    df = pd.read_csv(p)

    return df


def infer_test(
        model=lambda x: x,
        frame: pd.DataFrame = pd.DataFrame(), 
        src_column: str = 'infer',
        batch_size: int = HF_MBART_CONFIG['batch_size'],
    ):
    """
    Infer on the testing data using a batch-processing model.

    Args:
        - model : function or model, anything that can take a batch input i.e `model(list_of_inputs)`
        - frame <pd.Dataframe> : Dataframe to pull information from, model is evaulated on column src_column
        - src_column <str> : the column to look for our inference data in DEFAULT = `infer`
        - batch_size <int> : number of inputs to process per batch (default: 16).

    Returns:
        - (pandas.DataFrame, str): Dataframe with predictions for each row in column `prediction`

    Usage:
        ```
        frame = infer_test(model, data_frame, src_column='test', file_prefix='my_model')

        ```
    """
    tqdm.pandas()
    try:
        predictions = []
        # Process in batches
        for i in tqdm(range(0, len(frame), batch_size), desc=f"Processing test file"):
            batch = frame[src_column][i:i + batch_size].tolist()
            batch_predictions = model(batch)  # Apply model to batch
            predictions.extend(batch_predictions)

        # Add predictions to DataFrame
        frame['prediction'] = predictions

        return frame
    except Exception as e:
        raise e


def format_frame_for_sub(
        lang:str,
        frame:pd.DataFrame, 
        column_conversions:dict = 
            {
            'source': 'text'
            }
    ):
    """ take in a dataframe, and language (optional column conversion dict) and produce a dataframe that can be exported for our task submissions

        Args:
            - lang<str> : the language code we want to use 
            - frame<pd.Dataframe>: pandas dataframe containing at least an `id` column
            - column_conversions<dict> : a dictionary for converting the frames columns into the ones needed for semeval

                **IMPORTANT**: make sure that you have conversions from your columns into the following columns:
                    `prediction` (your predicted result) [if not already present in your df]

                    `text` (the source information) [if not already present in your df]

        Returns: 
            A dataframe with the following columns `['id', 'source_language', 'target_language', 'text', 'prediction']`

        Usage:
        ```
            from output_module import format_frame_for_sub

            # some code that placed predictions into a column 'test' into a df called DF

            out_frame = format_frame_for_sub('de', DF, column_conversion={'test': 'prediction', 'source': 'text'})

            out_frame.to_json(some_path.jsonl, orient='records', lines=True) #the last two args are important for making sure the json is the right format
        ```
    """

    frame['source_language'] = ['en']*len(frame)
    frame['target_language'] = [lang]*len(frame)

    for key, converted in column_conversions.items():
        frame[converted] = frame[key]

    
    desired_columns = ['id', 'source_language', 'target_language', 'text', 'prediction']

    return frame[desired_columns]