import gin

from econ2vec.inference import Econ2VecInferencer
from econ2vec.trainer import Econ2VecTrainer

gin.parse_config_file('config.gin')

trainer = Econ2VecTrainer()
# trainer.train()

inferencer = Econ2VecInferencer()
# inferencer.sim()
