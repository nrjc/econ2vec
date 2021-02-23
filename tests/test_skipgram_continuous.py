import unittest

import torch

from econ2vec.model import SkipGramContinuousModel


class TestSkipGramContinuous(unittest.TestCase):
    def test_skipgram_continuous_forward(self):
        embed_size = 20
        embed_dim = 10
        batch_size = 4
        # 1 vec, 5 neighbor
        center_vector = torch.rand((batch_size, 1, embed_size))
        neighbor_vector = torch.rand((batch_size, 5, embed_size))
        model = SkipGramContinuousModel(embed_size, embed_dim)
        loss = model.forward(center_vector, neighbor_vector)
        # Asserting loss is scalar
        self.assertEqual(loss.shape, torch.Size([]))


if __name__ == '__main__':
    unittest.main()
