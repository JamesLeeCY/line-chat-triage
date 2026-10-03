"""
Pluggable text classifiers for community stance / topic.

Every classifier exposes `fit(texts, labels)`, `predict_proba(texts)` and
`classes_`, so evaluation and the downstream entropy pipeline don't care
whether the model is TF-IDF, Jev, or anything else.
"""
import re
from typing import Protocol, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class TextClassifier(Protocol):
    classes_: Sequence[str]

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> "TextClassifier": ...

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray: ...


_URL = re.compile(r"https?://\S+")
_SPACES = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lower-case Latin, collapse URLs to one token and whitespace to one space."""
    return _SPACES.sub(" ", _URL.sub(" URL ", text)).strip().lower()


class TfidfLogReg:
    """
    Character n-gram TF-IDF + logistic regression.

    Char n-grams need no Chinese word segmentation and still catch slang
    (噴, 歐印, 套牢), tickers (2330, nvda) and emoji. `class_weight="balanced"`
    keeps the minority stance from being drowned by "neutral".
    """

    def __init__(self, ngram_range=(1, 3), min_df: int = 2, C: float = 4.0, max_features: int = 200_000):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                analyzer="char_wb", ngram_range=ngram_range, min_df=min_df,
                sublinear_tf=True, max_features=max_features, preprocessor=normalize,
            )),
            ("clf", LogisticRegression(C=C, max_iter=3000, class_weight="balanced")),
        ])

    @property
    def classes_(self) -> list[str]:
        return list(self.pipeline.named_steps["clf"].classes_)

    def fit(self, texts, labels):
        self.pipeline.fit(list(texts), list(labels))
        return self

    def predict_proba(self, texts) -> np.ndarray:
        return self.pipeline.predict_proba(list(texts))

    def predict(self, texts) -> list[str]:
        classes = self.classes_
        return [classes[i] for i in self.predict_proba(texts).argmax(axis=1)]

    def top_features(self, k: int = 15) -> dict[str, list[str]]:
        """The n-grams pushing hardest toward each class — a sanity check on what was learned."""
        vocab = np.array(self.pipeline.named_steps["tfidf"].get_feature_names_out())
        coef = self.pipeline.named_steps["clf"].coef_
        if coef.shape[0] == 1:  # binary: one row, negative = first class
            coef = np.vstack([-coef[0], coef[0]])
        return {cls: vocab[np.argsort(row)[::-1][:k]].tolist() for cls, row in zip(self.classes_, coef)}


class MajorityClass:
    """Baseline: always predicts the training-set class distribution."""

    def fit(self, texts, labels):
        values, counts = np.unique(list(labels), return_counts=True)
        self.classes_ = values.tolist()
        self._prior = counts / counts.sum()
        return self

    def predict_proba(self, texts) -> np.ndarray:
        return np.tile(self._prior, (len(texts), 1))
