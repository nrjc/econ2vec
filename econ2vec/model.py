import gin
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as f

"""
    u_embedding: Embedding for center vector.
    v_embedding: Embedding for neighbor vectors.
"""


@gin.configurable
class SkipGramContinuousModel(nn.Module):
    def __init__(self, emb_size, emb_dimension, neighbor_dim):
        super(SkipGramContinuousModel, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.u_embeddings = nn.Linear(emb_size, emb_dimension)
        self.v_embeddings = nn.Linear(emb_size, emb_dimension)
        self.v_compress_embed = nn.Conv1d(neighbor_dim * 2, 1, 1)

        initrange = 1.0 / self.emb_dimension
        init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
        init.uniform_(self.v_embeddings.bias.data, -initrange, initrange)
        init.uniform_(self.v_compress_embed.weight.data, -initrange, initrange)
        self.ts2id = dict()
        self.id2ts = dict()

    def forward(self, pos_u, pos_v):
        emb_u = self.u_embeddings(pos_u)
        emb_v = self.v_embeddings(pos_v)
        emb_v_comp = self.v_compress_embed(emb_v)
        raw_mult = torch.sum(torch.mul(f.normalize(emb_u, p=2, dim=2), f.normalize(emb_v_comp, p=2, dim=2)), dim=[1, 2])
        return torch.norm(1 - raw_mult, p=2)

    def set_id2ts(self, id2ts):
        self.id2ts = id2ts

    def save_embedding(self, file_name):
        embeddings = self.u_embeddings.weight.cpu().data.numpy()
        with open(file_name, 'w') as f:
            f.write(f'{self.emb_size} {self.emb_dimension}\n')
            for id, ts in self.id2ts.items():
                e = ' '.join(map(lambda x: str(x), embeddings[:, id]))
                f.write(f'{ts} {e}\n')

    def load_embedding(self, file_name):
        with open(file_name, 'r') as f:
            first_line = f.readline().split()
            self.__init__(int(first_line[0]), int(first_line[1]))
            u_embeddings = np.empty([self.emb_dimension, 0], dtype=float)
            id = 0
            for l in f:
                line = l.split()
                stock, emb = line[0], line[1:]
                self.ts2id[stock] = id
                self.id2ts[id] = stock
                id += 1
                emb = np.array(emb).astype(np.float)
                u_embeddings = np.column_stack((u_embeddings, emb))
        with torch.no_grad():
            self.u_embeddings.weight.copy_(torch.from_numpy(u_embeddings))
