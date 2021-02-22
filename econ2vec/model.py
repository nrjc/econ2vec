import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

"""
    u_embedding: Embedding for center vector.
    v_embedding: Embedding for neighbor vectors.
"""


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

    def forward(self, pos_u, pos_v):
        emb_u = self.u_embeddings(pos_u)
        emb_v = self.v_embeddings(pos_v)
        score = torch.sum(torch.mul(emb_u, emb_v), dim=1)
        score = torch.clamp(score, max=10, min=-10)
        score = -F.logsigmoid(score)
        return torch.mean(score)

    def save_embedding(self, id2word, file_name):
        embedding = self.u_embeddings.weight.cpu().data.numpy()
        with open(file_name, 'w') as f:
            f.write('%d %d\n' % (len(id2word), self.emb_dimension))
            for wid, w in id2word.items():
                e = ' '.join(map(lambda x: str(x), embedding[wid]))
                f.write('%s %s\n' % (w, e))
