# repo: https://github.com/dsfsi/textaugment
from textaugment import Wordnet, EDA

tw = Wordnet()
te = EDA()


def synonym_replacement_augmentation(text):
    return te.synonym_replacement(text)

def synonym_insertion_augmentation(text):
    return te.random_insertion(text)
