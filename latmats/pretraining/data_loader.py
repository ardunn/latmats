import os
import pickle
import json

import tqdm

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/")


def open_pickle(filename):
    """
    Load a pickle file from a filename.

    Args:
        filename (str): The absolute path filename of the pickle file to be loaded.

    Returns:
        The object from the pickle file.
    """
    with open(filename, "rb") as f:
        loaded = pickle.load(f)
    return loaded


def load_file(relative_filename: str, jsonify: bool =True, quiet: bool = False, limit: int = None):
    """
    Load a file by relative filename (e.g., material2index).

    Args:
        relative_filename (str): The relative filename
        jsonify (bool): Interpret text file lines as json (required for processed_abstracts)
        quiet (bool): Print when starting and done loading the file.

    Returns:
        The object from the file
    """

    filename = os.path.join(DATA_DIR, relative_filename)

    if not quiet:
        print(f"loading {filename}...")
    if filename.endswith(".txt"):
        with open(filename) as f:
            if jsonify:
                lines = f.readlines()
                loaded = []

                lines_limited = lines[:limit] if limit else lines
                for l in tqdm.tqdm(lines_limited):
                    loaded.append(json.loads(l))
            else:
                loaded = f.read()

    elif filename.endswith(".pkl"):
        loaded = open_pickle(filename)
    if not quiet:
        print(f"loaded {filename}")

    return loaded
