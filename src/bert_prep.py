from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

import os

TRAIN_FILEPATH = "train"


texts = []
labels = []

for (dirpath, dirnames, filenames) in os.walk(TRAIN_FILEPATH):
    for filename in filenames:
        if filename.endswith(".txt"):
            with open(os.path.join(dirpath, filename), "r") as f:
                lines = f.readlines()
                label = ""
                text = ""
                for idx, line in enumerate(lines):
                    if idx == 0:
                        label = int(line)
                    else:
                        text += (line + " ")
                text = text.replace("\n", " ")
                texts.append(text)
                labels.append(label)

texts = np.array(texts)
labels = np.array(labels)

data = {"text": list(texts), "label": list(labels)}

pd.DataFrame.from_dict(data).to_csv(os.path.join("data", "data.csv"))

X_train, X_test, y_train, y_test = train_test_split(texts, labels,
                                                   test_size=0.1, stratify=labels)

for (X, Y, dataset_type) in [(X_train, y_train, "train"), (X_test, y_test,
                                                           "test")]:
    data = {"text": list(X), "label": list(Y)}
    print(set(Y))
    pd.DataFrame.from_dict(data).to_csv(os.path.join("data/bert",
                                                     dataset_type + ".csv"))