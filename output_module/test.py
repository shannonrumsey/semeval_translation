import unittest
import torch
import os
import pandas

from model_outputs import Decoder
from evaluation_output import infer_test

class DecoderTests(unittest.TestCase):

    def test_decode(self):
        d = Decoder()
        mock_data = [[9,10]]
        mock_data = torch.Tensor(mock_data).int()

        out = d.decode_output(mock_data)

        self.assertEqual(out, [' importado'])

class InferenceTests(unittest.TestCase):

    def test_inference_func(self):
        # Test to ensure a) file is made, b) output is an instance of a df, and c) df contains proper columns required in output
        mock_model = lambda x: x
        infered_frame, out_path = infer_test(mock_model, lang='ar', file_prefix='test')

        self.assertTrue(os.path.exists(out_path), 'test file not created')
        self.assertIsInstance(infered_frame, pandas.DataFrame, 'Output was not a dataframe')
        self.assertListEqual(['id', 'source_language', 'target_language', 'text', 'prediction'], list(infered_frame.columns), 'Incorrect columns created')


if __name__ == '__main__':
    unittest.main()