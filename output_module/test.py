import unittest
import torch
import os
import pandas

from model_outputs import Decoder
from evaluation_output import infer_test, load_ent_test_info, load_test_dataframe, format_frame_for_sub

class DecoderTests(unittest.TestCase):

    def test_decode(self):
        d = Decoder()
        mock_data = [[9,10]]
        mock_data = torch.Tensor(mock_data).int()

        out = d.decode_output(mock_data)

        self.assertEqual(out, [' importado'])

class InferenceTests(unittest.TestCase):

    def test_load_frame(self):
        # test that we load in the right dataframe with columns 
        # id, source
        df = load_test_dataframe('ar')
        req_cols = set(['id', 'wikidata_id', 'entity_types', 'source', 'targets', 'source_locale', 'target_locale'])

        self.assertSetEqual(req_cols, set(df.columns), 'Required columns not present in dataframe')

    def test_load_ent(self):
        ent_df = load_ent_test_info('ar')
        req_cols = set(['source', 'target'])

        self.assertSetEqual(req_cols, set(ent_df.columns), 'Required columns not present in dataframe')        

    def test_format_out_frame(self):
        df = load_test_dataframe('ar')
        df['prediction'] = df['source']

        formated = format_frame_for_sub(lang='ar', frame=df)
        req_cols = set(['id', 'source_language', 'target_language', 'text', 'prediction'])

        self.assertSetEqual(req_cols, set(formated.columns), 'Required columns not present in dataframe')

    def test_infer(self):
        df = load_test_dataframe('ar')
        test_frame = infer_test(frame=df, src_column='source', batch_size=16)

        self.assertIn('prediction', test_frame.columns, 'Predictions not present in output frame')
        

if __name__ == '__main__':
    unittest.main()