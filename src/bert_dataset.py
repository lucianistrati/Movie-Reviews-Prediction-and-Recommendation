"""Parliamentary questions classification task """

import csv

import datasets

from datasets.tasks import TextClassification


_DESCRIPTION = """-"""

_CITATION = """-"""

_TRAIN_DOWNLOAD_URL = "-"
_TEST_DOWNLOAD_URL = "-"


class MovieReviewsDataset(datasets.GeneratorBasedBuilder):

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(
                        names=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                }
            ),
            homepage="http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html",
            citation=_CITATION,
            task_templates=[
                TextClassification(text_column="text", label_column="label",
                                   labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])],
        )

    def _split_generators(self, dl_manager):
        train_path = "data/train.csv"
        test_path = "data/test.csv"
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN,
                                    gen_kwargs={"filepath": train_path}),
            datasets.SplitGenerator(name=datasets.Split.TEST,
                                    gen_kwargs={"filepath": test_path}),
        ]

    def _generate_examples(self, filepath):
        """Generate Semantical documents examples."""
        class_to_idx_dict = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7,
                             9: 8, 10: 9}

        with open(filepath, encoding="utf-8") as csv_file:
            csv_reader = csv.reader(
                csv_file, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL,
                skipinitialspace=True
            )
            for id_, row in enumerate(csv_reader):
                _, text, label = row
                if label not in class_to_idx_dict.keys():
                    continue
                label = class_to_idx_dict[label]
                yield id_, {"text": text, "label": label}