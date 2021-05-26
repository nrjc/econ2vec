import gin
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

"""
    u_embedding: Embedding for center vector.
    v_embedding: Embedding for neighbor vectors.
"""

@gin.configurable
class SkipGramContinuousModel(nn.Module):
    def __init__(self, emb_size, emb_dimension):
        super(SkipGramContinuousModel, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.u_embeddings = nn.Linear(emb_size, emb_dimension)
        self.v_embeddings = nn.Linear(emb_size, emb_dimension)

        initrange = 1.0 / self.emb_dimension
        init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
        init.constant_(self.v_embeddings.weight.data, 0)

        self.ts2id = dict()
        self.id2ts = dict()

    def forward(self, pos_u, pos_v):
        emb_u = self.u_embeddings(pos_u)
        emb_v = self.v_embeddings(pos_v)
        score = torch.sum(torch.mul(emb_u, emb_v), dim=1)
        score = torch.clamp(score, max=10, min=-10)
        score = -F.logsigmoid(score)
        return torch.mean(score)

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
