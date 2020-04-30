import os
import pickle


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data")

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


def load_file(relative_filename: str):
    """
    Load a file by relative filename (e.g., material2index).

    Args:
        relative_filename (str): The relative filename

    Returns:
        The object from the file
    """

    filename = os.path.join(DATA_DIR, relative_filename)

    print(f"loading {filename}...")
    if filename.endswith(".txt"):
        with open(filename) as f:
            loaded = f.read()
    elif filename.endswith(".pkl"):
        loaded = open_pickle(filename)
    print(f"loaded {filename}")

    return loaded
