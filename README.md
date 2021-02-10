# econ2vec
First draft cribbed from https://github.com/Andras7/word2vec-pytorch
Fast word2vec implementation at competitive speed compared with fasttext. The slowest part is the python data loader. Indeed, Python isn't the fastest programming language, maybe you can improve the code :)

## Advantages

* Easy to understand, solid code
* Easy to extend for new experiments
* You can try advanced learning optimizers, with new learning technics
* GPU support

## Supported features

* Skip-gram
* Batch update
* Cosine Annealing
* Negative Sampling
* Sub-sampling of frequent word
