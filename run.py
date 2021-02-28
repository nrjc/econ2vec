import gin

from econ2vec.trainer import Word2VecTrainer

gin.parse_config_file('config.gin')
trainer = Word2VecTrainer()
trainer.train()
