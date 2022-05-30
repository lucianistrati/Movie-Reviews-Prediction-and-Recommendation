import nltk

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

import gensim
model = gensim.models.fasttext.load_facebook_model(
    'data/embeddings/cc.en.300.bin.gz')
