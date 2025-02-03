import json
import os
import torch

path = lambda x: os.path.join(os.path.dirname(__file__), x) #lambda go brrrrrr

DEFAULT_VOCAB = '../vocab/first_inv_vocab.json'
WHITESPACE_CHAR = '\u2581'

class Decoder():
    def __init__(self, inverse_vocab:str=DEFAULT_VOCAB):
        """
            Decoder class to decode model outputs

            Args:
                - inverse_vocab<str>: this file will be taken relative to this file 

            If file is not found will set inverse_vocab to an empty dict
            if another unaccounted error happens initialization will raise an exception
        """
        self.inv_vocab_file = path(inverse_vocab)

        try:
            with open(self.inv_vocab_file, 'r') as f: #load the vocab
                self.inv_vocab = json.loads(f.read())
        except FileNotFoundError:
            self.inv_vocab = {}
            print(f'Provided Inverse Vocab not found at path {self.inv_vocab_file}')
        except Exception as e:
            raise e
            
    def decode_output(self, outputs)->list:
        """
            Decode a batch of outputs

            Args:
                - outputs<tensor>: Should be a tensor of shape (batch_size, seq_len, vocab_size) i.e [ [id, id, id, ....], ....]

            Returns:
                A list of decoded outputs with removed whitespace of size batch_size

            Raises:
                - an exception if it encounters one 

        """
        out = ['']*len(outputs) #fix size to avoid memory alloc for append
        try:
            for i, item in enumerate(outputs): #iterate over each item in the batch
                tokens = [self.inv_vocab.get(f'{idx}', '') for idx in item] #get the tokens from the ids
                tokens = ''.join(tokens) #join on empty string
                tokens = tokens.replace(WHITESPACE_CHAR, ' ') #replace _ with whitespace? not sure if this is probelmatic for other langs
                out[i] = tokens
            return out
        except Exception as e:
            raise e #raise error out
