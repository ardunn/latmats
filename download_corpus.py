import pymongo
import os
from pprint import pprint
from pymatgen import Composition
import pymatgen
import json
import re
from gensim.utils import to_unicode, deaccent
from chemdataextractor.doc import Paragraph
import string


client = pymongo.MongoClient(os.getenv("MATSCHOLAR_STAGING_HOST"), username=os.getenv("MATSCHOLAR_STAGING_USER"),
                             password=os.getenv("MATSCHOLAR_STAGING_PASS"), authSource=os.getenv("MATSCHOLAR_STAGING_DB"))
db = client['matscholar_staging']


PAT_ALPHABETIC = re.compile(r'[a-zA-Z]+[\w\-()]*[\w\-]+', re.UNICODE)


def custom_tokenize(text, lowercase=False, deacc=False, encoding='utf8', errors="strict", to_lower=False, lower=False, cde=True):
    text = to_unicode(text, encoding, errors=errors)
    lowercase = lowercase or to_lower or lower
    if lowercase:
        text = text.lower()
    if deacc:
        text = deaccent(text)
    if cde:
        text = " ".join(text.split())
        cde_p = Paragraph(text)
        tokens = cde_p.tokens
        toks = []
        for sentence in tokens:
            toks.append([])
            for tok in sentence:
                if tok.text not in string.punctuation:
                    yield tok.text
    else:
        for match in PAT_ALPHABETIC.finditer(text):
            yield match.group()


def tokenize(content, token_min_len=2, token_max_len=80, lower=False):
    return [
        to_unicode(token) for token in custom_tokenize(content, lower=lower, errors='ignore')
        if token_min_len <= len(token) <= token_max_len and not token.startswith('_')
    ]


pipeline = []
# pipeline.append({"$limit": 1000})
pipeline.append({"$lookup": {
    "from": "entries_review",
    "localField": "doi",
    "foreignField": "doi",
    "as": "abstracts"}})
pipeline.append(
    {"$unwind": {"path": "$abstracts", "preserveNullAndEmptyArrays": True}})
pipeline.append({"$lookup": {
    "from": "relevance_vb",
    "localField": "doi",
    "foreignField": "doi",
    "as": "relevances"}})
pipeline.append(
    {"$unwind": {"path": "$relevances", "preserveNullAndEmptyArrays": True}})
pipeline.append({"$project": {
    "doi": 1,
    "abstract": "$abstracts.abstract",
    "MAT_summary": 1,
    "relevance": "$relevances.relevance",
    "_id": 0}})
abstracts = 0
elements = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Pa", "Al", "Np", "Am", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
            "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "U", "Pu", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"]

i_count = 0

with open("rel_abstracts_with_mats.txt", 'w') as f:
    with open('rel_abstracts.txt', 'r') as g:
        for l in g:
            entry = json.loads(l.strip())
            if i_count % 10000 == 0:
                print(i_count)
                print(abstracts)
                print("")
            i_count += 1
            try:
                abstract = dict()
                successful_parsing = True
                if not entry['relevance']:
                    successful_parsing = False
                mats = []
                for mat in entry['MAT_summary']:
                    try:
                        if all(
                                [str(e) in elements for e in Composition(mat).elements]):
                            mats.append(mat)
                    except (pymatgen.core.composition.CompositionError, ValueError):
                        pass
                if len(mats) == 0:
                    successful_parsing = False
                abstract['mats'] = mats
                abstract['abstract'] = entry['abstract']
                abstract['doi'] = entry['doi']
                abstract['tokenized_abstract'] = " ".join(
                    tokenize(entry['abstract']))
                if successful_parsing:
                    f.write(json.dumps(abstract) + "\n")
                    abstracts += 1
            except:
                pass
