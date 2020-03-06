import h5py
# from vocab_utils import create_bpe_vocab, bpe_to_index, tokenize_corpus
import pickle
import sys
import os
from numpy.random import shuffle
import numpy as np
import tensorflow as tf
import itertools
import collections
import re
from collections import defaultdict

rel_abstracts_with_mats_path = "rel_abstracts_with_mats.txt"

materials_list_path = "materials_list.txt"

rel_abstracts_file = "rel_abstracts.bpe"

vocab_file = "complete_vocab.bpe"


def write_materials_list():
    materials = set()
    with open(rel_abstracts_with_mats_path, 'r') as f:
        for l in f:
            materials_here = json.loads(l)['mats']
            for mat in materials_here:
                materials.add(mat)

    with open(materials_list_path, 'w') as f:
        for mat in materials:
            f.write(mat + "\n")

    with open("materials.txt", 'w') as f:
        for mat in materials:
            f.write(mat + " ")


def write_plain_tokenized_corpus():

    materials = set()
    with open(materials_list_path, 'r') as f:
        for line in f:
            materials.add(line.strip())

    def lower_exclude_mats(word):
        if word in materials:
            return word
        else:
            return word.lower()

    with open(rel_abstracts_with_mats_path, 'r') as f:
        with open(plain_tokenized, 'w') as g:
            for l in f:
                abstract = json.loads(l)['tokenized_abstract']
                abstract = [lower_exclude_mats(w) for w in abstract.split()]
                abstract = " ".join(abstract)
                g.write(abstract + "\n")


elements = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Pa", "Al", "Np", "Am", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
            "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "U", "Pu", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"]

formulare = re.compile(r'([A-Z][a-z]*)(\d*)')


def parse_formula(formula):
    pairs = formulare.findall(formula)
    length = sum((len(p[0]) + len(p[1]) for p in pairs))
    assert length == len(formula)
    formula_dict = defaultdict(int)
    for el, sub in pairs:
        formula_dict[el] += float(sub) if sub else 1

    keys = formula_dict.keys()
    values = formula_dict.values()
    total = float(sum(values))
    formula_vec = np.zeros((len(elements)))
    for k in keys:
        formula_vec[elements.index(k)] = formula_dict[k] / total
    return formula_vec


def write_bpe_vocab():
    word2index = dict()
    counter = collections.Counter()

    materials = set()
    with open(materials_list_path, 'r') as f:
        for line in f:
            try:
                parse_formula(line.strip())
                materials.add(line.strip())
            except (KeyError, AssertionError, ValueError, ZeroDivisionError):
                pass

    material2index = dict()
    for i, material in enumerate(materials):
        material2index[material] = i

    index2material = {v: k for k, v in material2index.items()}

    pickle.dump(material2index, open(os.path.join('material2index.pkl'), 'wb'))
    pickle.dump(index2material, open(os.path.join('index2material.pkl'), 'wb'))

    print(len(material2index.keys()))
    word2index["<PAD>"] = 0
    i_word = 1
    with open(vocab_file, 'r') as f:
        for l in f:
            word = l.split()[0]
            word2index[word] = i_word
            i_word += 1
    index2word = {v: k for k, v in word2index.items()}

    print(len(word2index.keys()))
    pickle.dump(word2index, open(os.path.join('word2index.pkl'), 'wb'))
    pickle.dump(index2word, open(os.path.join('index2word.pkl'), 'wb'))


# def write_indexed_corpus():
#     with open()


def write_train_tfrecord():
    # num_examples =
    validation_fraction = 0.0001

    word2index = pickle.load(
        open(os.path.join("data", 'word2index.pkl'), 'rb'))
    index2word = pickle.load(
        open(os.path.join("data", 'index2word.pkl'), 'rb'))

    vocab_size = len(word2index.items())

    def file_len(fname):
        with open(fname) as f:
            for i, l in enumerate(f):
                pass
        return i + 1

    n_titles = file_len(titles_train)
    n_titles_validation = int(file_len(titles_train) * validation_fraction)
    n_titles_train = file_len(titles_train) - n_titles_validation

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def serialize_train_example(abstract, title, title_out):
        """
        Creates a tf.Example message ready to be written to a file.
        """
        # Create a dictionary mapping the feature name to the tf.Example-compatible
        # data type.
        feature = {
            'abstract': _int64_feature(abstract),
            'title': _int64_feature(title),
            'titlte_out': _int64_feature(title_out)
        }

        # Create a Features message using tf.train.Example.

        example_proto = tf.train.Example(
            features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    def clean_index_pad(title, abstract):
        abstract_clean = abstract.strip()
        abstract_indexed = bpe_to_index(
            abstract_clean, word2index, abstract_maxlen)

        abstract_padded = np.zeros(
            (abstract_maxlen), dtype=int)
        # 2 = <PAD>
        abstract_padded[:] = 2
        abstract_padded[:len(abstract_indexed)
                        ] = abstract_indexed

        title_clean = title.strip()
        title_indexed = bpe_to_index(
            title_clean, word2index, title_maxlen)

        title_padded = np.zeros((title_maxlen), dtype=int)
        title_out_padded = np.zeros((title_maxlen), dtype=int)
        # 2 = <PAD>
        title_padded[:] = 2
        title_out_padded[:] = 2

        title_padded[:len(title_indexed)] = title_indexed

        # Out: shift target by one
        title_out_padded[:len(title_indexed) -
                         1] = title_indexed[1:]

        return abstract_padded, title_padded, title_out_padded

    def tf_example_generator_factory(start, stop):
        def tf_example_generator():
            # Generator for serialized example messages from our dataset
            with open(titles_train, 'r') as titles_file:
                with open(abstracts_train, 'r') as abstracts_file:
                    for title, abstract in itertools.islice(zip(titles_file, abstracts_file), start, stop):
                        yield serialize_train_example(*clean_index_pad(title, abstract))
        return tf_example_generator

    serialized_features_dataset = tf.data.Dataset.from_generator(
        tf_example_generator_factory(
            start=0, stop=n_titles_train), output_types=tf.string, output_shapes=())

    writer = tf.data.experimental.TFRecordWriter(tfrecord_filename)
    writer.write(serialized_features_dataset)

    serialized_features_dataset = tf.data.Dataset.from_generator(
        tf_example_generator_factory(
            start=n_titles_train, stop=n_titles), output_types=tf.string, output_shapes=())

    writer = tf.data.experimental.TFRecordWriter(tfrecord_validation_filename)
    writer.write(serialized_features_dataset)


def main():
    if sys.argv[1] == "write_materials_list":
        write_materials_list()
    elif sys.argv[1] == "write_bpe_vocab":
        write_bpe_vocab()
    elif sys.argv[1] == "write_train_tfrecord":
        write_train_tfrecord()
    elif sys.argv[1] == "write_plain_tokenized_corpus":
        write_plain_tokenized_corpus()


if __name__ == "__main__":
    main()
