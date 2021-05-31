import gin

from econ2vec.inference import Econ2VecInferencer

gin.parse_config_file('config_infer.gin')

inferencer = Econ2VecInferencer()
print(inferencer.cos_sim('GOOG', "GOOGL"))