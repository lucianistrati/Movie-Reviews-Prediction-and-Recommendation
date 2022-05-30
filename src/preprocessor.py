from autocorrect import Speller

from nltk.corpus import stopwords


import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from gensim.parsing.preprocessing import STOPWORDS

import string
import re


def autocorrect_text(text):
    check = Speller(lang='en')
    return check(text)


def clean_text(text):
    text = re.sub('<.*?>', ' ', text)

    text = text.translate(str.maketrans(' ', ' ', string.punctuation))

    text = re.sub('[^a-zA-Z]', ' ', text)

    text = re.sub("\n", " ", text)

    return text

def get_stopwords_set():
    nltk_sws = set(stopwords.words('english'))
    spacy_sws = set(STOP_WORDS)
    gensim_sws = set(STOPWORDS)

    return nltk_sws.union(spacy_sws).union(gensim_sws)

