from latmats.tasks.analogy_testing.data import load_analogies
from latmats.pretraining.data_loader import load_file
from latmats.pretraining.model import Word2VecPretrainingModel
from latmats.utils import example_generator_not_material

from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors
from mat2vec.processing import MaterialsTextProcessor
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from pymatgen import Composition
import tqdm
import pprint
import copy
from collections import OrderedDict

word2index = load_file("word2index_3mil.pkl")
mat2index = load_file("material2index.pkl")

# print("AsIn" in word2index)
# print("AsIn" in mat2index)
# print("InAs" in word2index)
# print("InAs" in mat2index)
#
w2vr = Word2VecPretrainingModel(name="dense-2-128", n_layers=2)
w2vr.compile()
w2vr.load_weights()
w2v = w2vr.model_word2vec
embedding_layer = w2v.layers[3]

mproc = MaterialsTextProcessor()

# THE WORD2INDEX ENTRIES ARE NORMALIZED, MUST USE THE NORMALIZED FORMULA
analogies, compound_list = load_analogies()

# for c in compound_list:
#     cn = mproc.normalized_formula(c)
#     if cn not in word2index:
#         print(f"{c}, {cn} not in word2index")
# NONE SHOULD BE NOT IN WORD2INDEX


compound_list = [mproc.normalized_formula(c) for c in compound_list]

print("generating word vector map")
word2vector = {word: embedding_layer(index) for word, index in word2index.items()}

print("normalizing all vectors")
word2vector = {word: vector/np.linalg.norm(vector) for word, vector in word2vector.items()}

# print(word2vector["the"])
# print(word2vector["the"].shape)
# print(np.linalg.norm(word2vector["the"]))


for analogy in analogies["compounds_structures"]:

    a = analogy["relation1"][0]
    b = analogy["relation1"][1]
    c = analogy["relation2"][0]
    d = analogy["relation2"][1]

    av = word2vector[a]
    bv = word2vector[b]
    cv = word2vector[c]
    dv = word2vector[d]

    test_vector = np.add(cv, np.subtract(bv, av))

    word2similarity = {w: 0 for w in word2vector.keys()}
    for w, v in word2vector.items():
        # cosine_similarities[i] = np.dot(v, test_vector) / (np.linalg.norm(test_vector) * np.linalg.norm(v))
        word2similarity[w] = np.dot(v, test_vector)

    word2similarity_tuples = [(w, v) for w, v in word2similarity.items()]
    sorted_word2simtuples = sorted(word2similarity_tuples, key=lambda word2simtuple: word2simtuple[1], reverse=True)
    sorted_words = [w[0] for w in sorted_word2simtuples]
    print(f"{a} is to {b} as {c} is to {d}: rank {sorted_words.index(d)} with similarity {word2similarity[d]}")


    # cosine_similarities_idx = np.argsort(cosine_similarities)
    # words_sorted = [words[i] for i in cosine_similarities_idx.tolist()]
    # target_word_idx = words_sorted.index(d)
    # print(f"{a} is to {b} as {c} is to {d}: rank {target_word_idx} with similarity {cosine_similarities[target_word_idx]}")





