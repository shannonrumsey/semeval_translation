import unittest
import torch

from model_outputs import Decoder

class DecoderTests(unittest.TestCase):

    def test_decode(self):
        d = Decoder()
        mock_data = [[9,10]]
        mock_data = torch.Tensor(mock_data).int()

        out = d.decode_output(mock_data)

        self.assertEqual(out, [' importado'])


if __name__ == '__main__':
    unittest.main()