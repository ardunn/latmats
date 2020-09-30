import pprint
import json

import tqdm
from pymatgen.core.composition import Composition, CompositionError
from mat2vec.processing import MaterialsTextProcessor

from latmats.pretraining.data_loader import load_file


TRAINING_CORPUS = "processed_abstracts_excluding_analogies.json"

def load_analogies(filename, quiet=True):
    """
    Rules:
    1. Only keep analogy sections relevant to compounds
    2. Remove analogies not containing a single compound (i.e., only elements)
    3. Remove analogies requiring a composition not in processed_abstracts


    Args:
        filename:

    Returns:

    """

    missing_compositions = [
        "AgF",
        "AsB",
        "AsTi",
        "BrRb",
        "CCoO3",
        'CoCr2O4',
        'CrF3',
        'F2Mn',
        'F2Ni',
        'F2V',
        'F2Zn',
        'FK',
        'FRb',
        'Fe3S4',
        'GIY',
        'GaMnN',
        'GdN',
        'HgSe',
        'IK',
        'IRb',
        'MnPt',
        'PTi'
    ]
    missing_compositions = missing_compositions + [Composition(c).reduced_formula for c in missing_compositions]

    with open(filename, "r") as f:
        analogy_list = f.readlines()

    section_headers = []
    analogies_by_section = []

    section_mapping = {
        ": crystal structures (zincblende, wurtzite, rutile, rocksalt, etc.)": "compounds_structures",
        ": crystal symmetry (cubic, hexagonal, tetragonal, etc.)": "compounds_symmetry",
        ": magnetic properties": "compounds_magnetic",
        ": metals and their oxides (most common)": "oxides"
    }
    analogies_by_this_section = []
    for a in analogy_list:
        if ":" in a:
            a_key = section_mapping[a.replace("\n", "")]
            section_headers.append(a_key)
            if analogies_by_this_section:
                analogies_by_section.append(analogies_by_this_section)
                analogies_by_this_section = []
            continue
        else:
            analogies_by_this_section.append(a.replace("\n", ""))
    else:
        analogies_by_section.append(analogies_by_this_section)

    analogies_raw = dict(zip(section_headers, analogies_by_section))
    analogies = {k: [] for k in analogies_raw.keys()}
    compound_list = []
    for alabel, aset in analogies_raw.items():
        for a in aset:
            split = a.split(" ")
            if len(split) != 4:
                raise ValueError
            analogy_set = {"relation1": [split[0], split[1]],
                           "relation2": [split[2], split[3]]}

            # remove unusable analogies as per this function doc rules
            species = []
            for i in analogy_set["relation1"] + analogy_set["relation2"]:
                try:
                    c = Composition(i)
                    species.append(c)
                except CompositionError:
                    pass

            if not all([len(s.elements) == 1 for s in species]) and species and \
                    not any([s.reduced_formula in missing_compositions for s in species]):

                analogies[alabel].append(analogy_set)
                for s in species:
                    if len(s.elements) > 1:
                        compound_list.append(s.reduced_formula)
            else:
                if not quiet:
                    print(f"Analogy {a} not included as per rules!")

    compound_list = set(compound_list)
    # print("MISSING", [c for c in compound_list if c in missing_compositions])

    if not quiet:
        n_analogies = sum([len(aset) for aset in analogies.values()])
        print(f"Extracted {n_analogies} analogies, containing {len(compound_list)} unique compounds.")

    return analogies, compound_list


def create_training_corpus(excluded_compounds):
    abstracts = load_file("processed_abstracts.txt", limit=None)
    mproc = MaterialsTextProcessor()
    excluded_compounds_normalized = [mproc.normalized_formula(ec) for ec in excluded_compounds]
    excluded_compounds_all_normalized_forms = list(zip(list(excluded_compounds), excluded_compounds_normalized))
    n_excluded = {ec: 0 for ec in excluded_compounds_normalized}
    clean_abstracts = []
    for a in tqdm.tqdm(abstracts):
        include_abstract = True
        for ec, ecn in excluded_compounds_all_normalized_forms:
            if ec in a or ecn in a:
                n_excluded[ecn] += 1
                include_abstract = False
        if include_abstract:
            clean_abstracts.append(a)

    print("Created training dataset without the following excluded compounds (normalized version shown):")
    pprint.pprint(n_excluded)
    # print(clean_abstracts)

    with open(TRAINING_CORPUS, "w") as f:
        json.dump(clean_abstracts, f)
    return clean_abstracts


def load_training_corpus():
    with open(TRAINING_CORPUS, "r") as f:
        c = json.load(f)
    return c


if __name__ == "__main__":
    analogies, compound_list = load_analogies("analogies_hiddenrep_test.txt", quiet=False)

    # pprint.pprint(compound_list)
    create_training_corpus(compound_list)


    # print(len(load_training_corpus()))


    # correct number of analogies is present
    # s = 0
    # for k, v in analogies.items():
    #     print(k, len(v))
    #     s += len(v)
    #
    # print(s)


    # pprint.pprint(analogies)
