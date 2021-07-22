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
        super().__init__()
        self.emb_dimension = emb_dimension
        self.emb_size = emb_size
        self.neighbor_dim = neighbor_dim
        self.u_embeddings = nn.Linear(emb_size, emb_dimension)
        self.u_compress_embed = nn.Conv1d(in_channels=1,
                                          out_channels=2 * neighbor_dim,
                                          groups=1, kernel_size=(1,))
        self.v_embeddings = nn.Linear(emb_dimension, emb_size)
        initrange = 1.0 / self.emb_dimension
        init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
        init.uniform_(self.v_embeddings.bias.data, -initrange, initrange)
        init.uniform_(self.u_compress_embed.weight.data, -initrange, initrange)
        self.ts2id = dict()
        self.id2ts = dict()

    def forward(self, pos_u, pos_v):
        """
        pos_u: *, 1, E
        pos_v: *, N, E
        *, 1, E -> *, 1, C -> *, N, E
        """
        # *, 1, E -> *, 1, C
        b1 = self.u_embeddings(pos_u)
        # *, 1, C -> *, N, C
        b2 = self.u_compress_embed(b1)
        # *, N, C -> *, N, E
        b3 = self.v_embeddings(b2)
        # L2 Norm Difference
        return torch.norm(b3 - pos_v, p=2)

    def compute_embedding(self, pos_u):
        return self.u_embeddings(pos_u)

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
