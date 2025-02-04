import os
import pandas as pd
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import semeval_langs


def infer_test(model = lambda x: x, lang:str = None, file_prefix:str = 'generic'):
    """
        Infer on the testing data by passing in a model (or some function)
        that can take in the input s.t model(input) -> translation 

        Args:
            - model : function or model, anything that can take an input i.e `model(inputs)`
            - lang<str> : the language we are translating i.e ['ar', 'de', 'ko', etc..]
            - file_prefix<str> : the prefix you would like to prepend to the file name
        Returns: 
            A tuple s.t,
                (pandas.DataFrame, string)
                containing the output frame and the output file path (absolute path)

        Usage:
        ```
        from output_module import infer_test

        # .. some model code 
        # given the var model can take in an input like model(input) -> output

        frame, out_file = infer_test(model, lang='ar', file_prefix='my_model')

        #frame will contain columns [id, source_language, target_language, text, prediction] which are used for eval 
        #out_file is the absolute path of the file generated
        
        #running on all languages
        for lang in ['ar', 'es', 'de', 'fr', 'it', 'ja', 'ko', 'th', 'tr', 'zh']:
            lang_frame, lang_file = infer_test(model, lang=lang, file_prefix='final')
    """
    file_name = semeval_langs[lang]
    #load test file 
    f_path = os.path.join(os.path.dirname(__file__), f'../data/semeval_test/{file_name}.jsonl')
    out_dir = os.path.join(os.path.dirname(__file__), '../outputs/')
    try:
        json_df = pd.read_json(f_path, lines=True)
        json_df['prediction'] = json_df['source'].apply(model) #apply model to text
        json_df['text'] = json_df['source']
        json_df['source_language'] = ['en'] * len(json_df['source'])
        json_df['target_language'] = [lang] * len(json_df['source'])

        os.makedirs(out_dir, exist_ok=True)
        desired_columns = ['id', 'source_language', 'target_language', 'text', 'prediction']
        out_path = f'{out_dir}/{file_prefix}_{lang}.jsonl'
        json_df[desired_columns].to_json(out_path, orient='records', lines=True)
        return json_df[desired_columns], out_path
    except Exception as e:
        raise e
    

