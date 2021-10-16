import gin

from econ2vec.inference import Econ2VecInferencer

gin.parse_config_file('config_infer.gin')

inferencer = Econ2VecInferencer()
print(f"cosine similarity between GOOG and GOOGL: {inferencer.cos_sim('GOOG', 'GOOGL')}")
print(f"cosine similarity between JPM and BAC: {inferencer.cos_sim('JPM', 'BAC')}")
print(f"cosine similarity between FB and BAC: {inferencer.cos_sim('FB', 'BAC')}")
print(f"cosine similarity between FB and GOOGL: {inferencer.cos_sim('FB', 'GOOGL')}")
print(f"cosine similarity between FB and AMZN: {inferencer.cos_sim('FB', 'AMZN')}")
print(f"cosine similarity between GBPUSD and GBPSGD: {inferencer.cos_sim('GBPUSD=X', 'GBPSGD=X')}")
print(f"cosine similarity between GBPUSD and AMZN: {inferencer.cos_sim('GBPUSD=X', 'AMZN')}")
