from dataclasses import dataclass, field
from typing import Any

import gin
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from econ2vec.data_grabber import YahooFinanceETL
from econ2vec.model import SkipGramContinuousModel


@gin.configurable
@dataclass
class Econ2VecTrainer:
    batch_size: int
    iterations: int
    initial_lr: float = 1e-3
    verbose: bool = False
    use_cuda: bool = torch.cuda.is_available()
    device: Any = None
    dataset: YahooFinanceETL = None
    model: SkipGramContinuousModel = None
    dataloader: DataLoader = None
    embedding_filename: str = "out.vec"
    weight_decay: float = 0

    def __post_init__(self):
        self.dataset = YahooFinanceETL()
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size,
                                     shuffle=False, num_workers=0, collate_fn=self.dataset.collate)
        self.model = SkipGramContinuousModel(emb_size=self.dataset.get_emb_size())
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        if self.use_cuda:
            self.model.cuda()

    def train(self):
        for iteration in range(self.iterations):
            print(f"Iteration: {(iteration + 1)}")
            optimizer = optim.Adam(self.model.parameters(), lr=self.initial_lr, weight_decay=self.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(self.dataloader))

            running_loss = 0.0

            for i, sample_batched in enumerate(tqdm(self.dataloader, disable=not self.verbose)):
                if len(sample_batched[0]) > 1:
                    pos_u = sample_batched[0].to(self.device)
                    neg_v = sample_batched[1].to(self.device)

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    loss = self.model.forward(pos_u, neg_v)
                    loss.backward()

                    running_loss = running_loss * 0.9 + loss.item() * 0.1
                    if i > 0 and i % 500 == 0:
                        print(f"Loss: {str(running_loss)}")
            print(running_loss)

        self.model.set_id2ts(self.dataset.id2ts)
        self.model.save_embedding(self.embedding_filename)