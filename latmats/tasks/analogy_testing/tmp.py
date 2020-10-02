from latmats.tasks.analogy_testing.data import load_analogies
from latmats.pretraining.data_loader import load_file
from latmats.pretraining.model import Word2VecPretrainingModel
from latmats.utils import example_generator_not_material

from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors
from mat2vec.processing import MaterialsTextProcessor
import numpy as np
import time

word2index = load_file("word2index_3mil.pkl")
mat2index = load_file("material2index.pkl")

w2vr = Word2VecPretrainingModel(name="dense-2-128", n_layers=2)
w2vr.compile()
w2vr.load_weights()
w2v = w2vr.model_word2vec
embedding_layer = w2v.layers[3]

mproc = MaterialsTextProcessor()


print("generating word vector map")
word2vector = {word: embedding_layer(index) for word, index in list(word2index.items())[:100]}

print([k for k in word2vector.keys()])

print("normalizing all vectors")
word2vector = {word: vector / np.linalg.norm(vector) for word, vector in word2vector.items()}


av = word2vector["the"]
bv = word2vector["of"]
cv = word2vector["and"]


words = list(word2vector.keys())
vectors = list(word2vector.values())

test_vector = np.add(cv, np.subtract(bv, av))

t0 = time.time()
cosine_similarities = [0] * len(vectors)
for i, v in enumerate(vectors):
    cosine_similarities[i] = np.dot(v, test_vector) / (np.linalg.norm(v) * np.linalg.norm(test_vector))
t1= time.time()
print("python native", t1-t0)

# cosine_similarities_wekv = WordEmbeddingsKeyedVectors.cosine_similarities(test_vector, vectors)

t2 = time.time()
norm = np.linalg.norm(test_vector)
all_norms = np.linalg.norm(vectors, axis=1)
dot_products = np.dot(vectors, test_vector)
similarities_wekv = dot_products / (norm * all_norms)
t3 = time.time()
print("numpy way", t3-t2)

print(cosine_similarities)
print(similarities_wekv)


