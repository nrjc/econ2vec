from dataclasses import dataclass
import gin
import torch
import torch.nn as nn

from econ2vec.model import SkipGramContinuousModel


@gin.configurable
@dataclass
class Econ2VecInferencer:
    skip_gram_model: SkipGramContinuousModel = None
    embedding_filename: str = "out.vec"

    def __post_init__(self):
        self.model = SkipGramContinuousModel(emb_size=1)
        self.model.load_embedding(file_name=self.embedding_filename)

    def ticker2emb(self, ticker):
        id = self.model.ts2id[ticker]
        emb = self.model.u_embeddings.weight.t()[id, :]
        return emb

    def cos_sim(self, ticker1, ticker2):
        emb1, emb2 = self.ticker2emb(ticker1), self.ticker2emb(ticker2)
        cos = nn.CosineSimilarity(dim=0, eps=1e-8)
        res = cos(emb1, emb2).item()
        return round(res, 3)
