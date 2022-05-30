import pandas as pd

import fasttext
import csv
import os

SAVING_FOLDER = "data"
def save_data(df, train_size, target_column, content_column):
    fasttext_labels_list = []
    for i in range(len(df)):
        fasttext_labels_list.append('__label__' + str(df.iloc[i][
            target_column]))

    df[target_column] = fasttext_labels_list

    df = df[[target_column, content_column]]
    # the column order needs to be changed for processing with the FastText
    # model
    df = df.reindex(columns=[target_column, content_column])

    df[:int(train_size * len(df))].to_csv(
        os.path.join(SAVING_FOLDER, "train.txt"),
        index=False,
        sep=' ',
        header=False,
        quoting=csv.QUOTE_NONE,
        quotechar="",
        escapechar=" ")

    df[int(train_size * len(df)):].to_csv(
        os.path.join(SAVING_FOLDER, "test.txt"),
        index=False,
        sep=' ',
        header=False,
        quoting=csv.QUOTE_NONE,
        quotechar="",
        escapechar=" ")


def train_fasttext_model(train_size, test_size):
    model = fasttext.train_supervised(input=os.path.join("data",
                                                         "train.txt"),
                                      autotuneValidationFile="data/test.txt")
    model.save_model(
        "fasttext_model_" +
        str(train_size) +
        "-" +
        str(test_size) +
        ".bin")
    return model


def train_fasttext(df, target_column, content_column):
    model_name = "Fasttext Model"
    train_size = 0.8
    test_size = 0.2
    save_data(df, train_size, target_column, content_column)
    # trains the fast-text model on the first (train_size * 100) % of the data
    model = train_fasttext_model(train_size, test_size)
    # tests the fast-text model accuracy on the last ((train_size -
    # test-size) * 100) % of the data


def fasttext_model():
    try:
        os.mkdir("data")
    except FileExistsError:
        pass
    df = pd.read_csv("data/data.csv")
    train_fasttext(df, "label", "text")


def main():
    fasttext_model()


if __name__ == '__main__':
    main()


"""
Fasttext 

Progress: 100.0% Trials:   15 Best score:  0.464000 ETA:   0h 0m 0s
Training again with best arguments
Read 2M words
Number of words:  153200
Number of labels: 10
Progress: 100.0% words/sec/thread:  490771 lr:  0.000000 avg.loss:  1.498224 ETA:   0h 0m 0s
"""