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
        self.skip_gram_model = SkipGramContinuousModel(emb_size=1)
        self.skip_gram_model.load_embedding(file_name=self.embedding_filename)
