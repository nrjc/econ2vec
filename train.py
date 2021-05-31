import gin

from econ2vec.trainer import Econ2VecTrainer

gin.parse_config_file('config_train.gin')

trainer = Econ2VecTrainer()
trainer.train()
