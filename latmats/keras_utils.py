import tensorflow as tf
import numpy as np
import math
import re
from collections import defaultdict
from tensorflow.keras.layers import Layer

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

    max_length = 9
    total = float(sum(formula_dict.values()))

    matrix = np.zeros((max_length, len(elements) + 1))
    for i, ent in enumerate(formula_dict.items()):
        el, num = ent
        el = elements.index(el)
        matrix[i, el] = num
        matrix[i, -1] = num / total

    return matrix


def penalized_tanh(x):
    alpha = 0.25
    return tf.maximum(tf.nn.tanh(x), alpha * tf.nn.tanh(x))
