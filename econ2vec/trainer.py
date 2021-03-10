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
class Word2VecTrainer:
    batch_size: int
    iterations: int
    initial_lr: float = 1e-3
    use_cuda: bool = torch.cuda.is_available()
    verbose: bool = False
    device: Any = None
    dataset: YahooFinanceETL = None
    skip_gram_model: SkipGramContinuousModel = None
    dataloader: DataLoader = None

    def __post_init__(self):
        self.dataset = YahooFinanceETL()
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size,
                                     shuffle=False, num_workers=0, collate_fn=self.dataset.collate)
        self.skip_gram_model = SkipGramContinuousModel()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        if self.use_cuda:
            self.skip_gram_model.cuda()

    def train(self):
        for iteration in range(self.iterations):
            print("Iteration: " + str(iteration + 1))
            optimizer = optim.Adam(self.skip_gram_model.parameters(), lr=self.initial_lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(self.dataloader))

            running_loss = 0.0

            for i, sample_batched in enumerate(tqdm(self.dataloader, disable=not self.verbose)):
                if len(sample_batched[0]) > 1:
                    pos_u = sample_batched[0].to(self.device)
                    neg_v = sample_batched[1].to(self.device)

                    scheduler.step()
                    optimizer.zero_grad()
                    loss = self.skip_gram_model.forward(pos_u, neg_v)
                    loss.backward()
                    optimizer.step()

                    running_loss = running_loss * 0.9 + loss.item() * 0.1
                    if i > 0 and i % 500 == 0:
                        print(" Loss: " + str(running_loss))
