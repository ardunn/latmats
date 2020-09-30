import pprint

from pymatgen.core.composition import Composition, CompositionError
from mat2vec.processing import MaterialsTextProcessor

from latmats.pretraining.data_loader import load_file


def load_analogies(filename, quiet=True):
    """
    Rules:
    1. Only keep analogy sections relevant to compounds
    2. Remove analogies not containing a single compound (i.e., only elements)


    Args:
        filename:

    Returns:

    """
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

            if not all([len(s.elements) == 1 for s in species]) and species:
                analogies[alabel].append(analogy_set)
                for s in species:
                    if len(s.elements) > 1:
                        compound_list.append(s.reduced_formula)
            else:
                if not quiet:
                    print(f"Analogy {a} not included as per rules!")
    compound_list = set(compound_list)

    if not quiet:
        n_analogies = sum([len(aset) for aset in analogies.values()])
        print(f"Extracted {n_analogies} analogies, containing {len(compound_list)} unique compounds.")

    return analogies, compound_list


def create_training_corpus(excluded_compounds):
    abstracts = load_file("processed_abstracts.txt", limit=1000)
    mproc = MaterialsTextProcessor()

    mproc.normalized_formula()


    return abstracts



if __name__ == "__main__":
    analogies, compound_list = load_analogies("analogies_hiddenrep_test.txt", quiet=False)

    abstracts = create_training_corpus(None)

    for a in abstracts:
        print(a)

    # pprint.pprint(analogies)
    # print(len(compound_list))


    # correct number of analogies is present
    # s = 0
    # for k, v in analogies.items():
    #     print(k, len(v))
    #     s += len(v)
    #
    # print(s)


    # pprint.pprint(analogies)
