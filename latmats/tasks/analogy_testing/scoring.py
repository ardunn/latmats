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


for analogy in analogies["element_names"]:

    a = analogy["relation1"][0]
    b = analogy["relation1"][1]
    c = analogy["relation2"][0]
    d = analogy["relation2"][1]

    av = word2vector[a]
    bv = word2vector[b]
    cv = word2vector[c]
    dv = word2vector[d]

    words = list(word2vector.keys())
    vectors = list(word2vector.values())

    test_vector = np.add(cv, np.subtract(bv, av))
    norm = np.linalg.norm(test_vector)

    cosine_similarities = [0] * len(vectors)
    for i, v in enumerate(vectors):
        v_norm = np.linalg.norm(v)
        cosine_similarities[i] = np.dot(vectors[i], test_vector) / (v_norm * norm)

    word2similarity = dict(zip(words, cosine_similarities))

    # print("computing similarities")
    # cosine_similarities = WordEmbeddingsKeyedVectors.cosine_similarities(test_vector, vectors)

    # print(word2similarity["ferromagnetic"])

    # print("sorting...")
    # limit = 10
    sorted_words = {w: word2similarity[w] for w in sorted(words, key=lambda w: word2similarity[w], reverse=True)}

    print(f"{a} is to {b} as {c} is to {d}: rank {list(sorted_words.keys()).index(d)}")


    # i = 1
    # for w, s in sorted_words.items():
    #     print(f"{i} {w}: {s}")
    #     if w == "ferromagnetic":
    #         break
    #     i += 1
