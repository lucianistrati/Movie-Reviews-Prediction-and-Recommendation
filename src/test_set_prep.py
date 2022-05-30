import pandas as pd

import os

TEST_FILEPATH = "test"

texts = []
for (dirpath, dirnames, filenames) in os.walk(TEST_FILEPATH):
    for filename in filenames:
        if filename.endswith(".txt"):
            with open(os.path.join(dirpath, filename), "r") as f:
                lines = f.readlines()
                text = ""
                for idx, line in enumerate(lines):
                   text += (line + " ")
                text = text.replace("\n", " ")
                texts.append(text)

data = {"text": texts}

pd.DataFrame.from_dict(data).to_csv(os.path.join(TEST_FILEPATH, "test.csv"))