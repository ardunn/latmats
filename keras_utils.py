import tensorflow as tf
import numpy as np
import re
from collections import defaultdict

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


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def is_material(word, material2index):
    try:
        parse_formula(word)
        if not word in elements:
            return 1.0
        else:
            return 0.0
    except (KeyError, AssertionError, ValueError, ZeroDivisionError):
        return 0.0


def fancy_sentence_sampler(sentence, window_size, material2index=None):
    sentence_length = len(sentence)
    if material2index is not None:
        mask = [is_material(w, material2index) for w in sentence]
        mask = mask / np.sum(mask)
        indices = np.arange(0, sentence_length)
        centre = np.random.choice(indices, p=mask)
    else:
        centre = np.random.randint(0, sentence_length)

    if centre <= window_size:
        min_value = 0
    else:
        min_value = centre - window_size
    if centre + window_size + 1 > sentence_length:
        max_value = sentence_length
    else:
        max_value = centre + window_size + 1

    indices = np.arange(min_value, max_value)
    probs = list(reversed(np.arange(1, centre - min_value + 1))) + \
        [0] + list(np.arange(1, max_value - centre))

    probs = softmax(probs)

    index = np.random.choice(indices, p=probs)

    context = sentence[index]
    target = sentence[centre]
    return target, context


def example_generator_not_material(corpus, window_size, word2index):
    n_examples = len(corpus)

    while True:
        abstract = corpus[np.random.randint(0, n_examples)]

        target, context = fancy_sentence_sampler(abstract, window_size)

        # Dummy output
        yield (word2index[target], word2index[context]), 0


def example_generator_material(corpus, window_size, word2index, material2index):
    n_examples = len(corpus)

    while True:
        while True:
            abstract = corpus[np.random.randint(0, n_examples)]
            if any([is_material(w, material2index) == 1 for w in abstract]):
                break

        target, context = fancy_sentence_sampler(
            abstract, window_size, material2index=material2index)

        yield (parse_formula(target), word2index[context]), 0
