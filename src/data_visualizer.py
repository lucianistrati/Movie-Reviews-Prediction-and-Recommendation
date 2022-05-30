from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk.tokenize import word_tokenize
from matplotlib import pyplot as plt
from collections import Counter
from preprocessor import get_stopwords_set

import numpy as np
import seaborn as sns
import pandas as pd

import statistics
import spacy
nlp = spacy.load("en_core_web_sm")

df = pd.read_csv("data/data.csv")


from copy import deepcopy

def plot_wordcloud(text, label, stopwords_set):
    # Create and genera te a word cloud image:
    wordcloud = WordCloud(stopwords=stopwords_set).generate(text)

    # Display the generated image:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Movie reviews labelled with " + str(label))
    plt.show()


stopwords_set = get_stopwords_set()
stopwords_set.update(["movie", "ta", "the", "this", "film", "hi", "ha",
                      "like", "character", "wa", "is", "time", "scene",
                      "story", "doe", "people", "It"])

for label in range(1, 11):
    selected_rows = deepcopy(df.loc[df['label'] == label])
    selected_rows.index = np.arange(0, len(selected_rows))
    documents_text = ""
    for i in range(len(selected_rows)):
        documents_text += (selected_rows.iloc[i]['text'] + " ")
    plot_wordcloud(documents_text, label, stopwords_set)


labels = dict(Counter(df['label'].tolist())).items()
score_freqs = [freq for (_, freq) in sorted(labels, key=lambda x:x[0])]

class_weight = [len(df) / freq for (_, freq) in sorted(labels,
                                                         key=lambda x:x[0])]
print("Class weights: ", class_weight)

sns.barplot(x=list(range(1, 11)), y=score_freqs)
plt.xlabel("Score")
plt.ylabel("Number of reviews")
plt.title("Reviews by scores")
plt.show()


precalculated = True


if precalculated:
    train_num_tokens = np.load("data/train_num_tokens.npy", allow_pickle=True)
    test_num_tokens = np.load("data/test_num_tokens.npy", allow_pickle=True)
else:
    train_num_tokens = []
    test_num_tokens = []

    train_df = pd.read_csv("train/train.csv")
    for i in range(len(train_df)):
        train_num_tokens.append(len(word_tokenize(train_df.iloc[i]['text'])))
    np.save("data/train_num_tokens.npy", np.array(train_num_tokens),
            allow_pickle=True)

    test_df = pd.read_csv("test/test.csv")
    for i in range(len(test_df)):
        test_num_tokens.append(len(word_tokenize(test_df.iloc[i]['text'])))
    np.save("data/test_num_tokens.npy", np.array(test_num_tokens),
            allow_pickle=True)


print("TRAIN max num. tokens: {} , min num. tokens: {}".format(max(
    train_num_tokens), min(train_num_tokens)))
print("TEST max num. tokens: {} , min num. tokens: {}".format(max(
    test_num_tokens), min(test_num_tokens)))


colors = ['r', 'b']
labels = ['Distribution of the number of tokens in the TRAIN reviews',
          'Distribution of the number of tokens in the TEST reviews']


fig, ax = plt.subplots()
for idx, data in enumerate([train_num_tokens, test_num_tokens]):
   sns.histplot(data, bins=250, ax=ax, kde=False, color=colors[idx],
                label=labels[idx])
ax.set_xlim([0, 2550])
plt.axvline(np.mean(train_num_tokens), label="TRAIN Mean number of tokens",
            color='m')
plt.axvline(np.mean(test_num_tokens), label="TEST Mean number of tokens",
            color='m')

plt.axvline(np.median(train_num_tokens), label="TRAIN Median number of tokens",
            color='g')
plt.axvline(np.median(test_num_tokens), label="TEST Median number of tokens",
            color='g')

plt.axvline(statistics.mode(train_num_tokens), label="TRAIN Mode number of tokens",
            color='c')
plt.axvline(statistics.mode(test_num_tokens), label="TEST Mode number of tokens",
            color='c')

plt.legend()
plt.show()


