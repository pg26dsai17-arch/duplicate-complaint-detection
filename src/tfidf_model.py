"""Model A: TF-IDF + cosine similarity (lexical baseline).

Each complaint becomes a sparse vector with one weight per word / word pair:
    tf(t,d)  = 1 + ln(count of t in d)
    idf(t)   = ln((1 + N) / (1 + df(t))) + 1
    weight   = tf * idf, then the vector is scaled to length 1.
Rare words get high weight. Two complaints are similar only if they SHARE words,
so a paraphrase with different words ("waste van" vs "garbage") scores low.
"""
from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocessing import normalize_for_tfidf


class TfidfModel:
    name = "tfidf"
    display_name = "TF-IDF"

    def __init__(self, ngram_range=(1, 2)):
        self.vectorizer = TfidfVectorizer(ngram_range=ngram_range, sublinear_tf=True)
        self.fitted = False

    def fit(self, texts):
        if len(texts) == 0:
            raise ValueError("Cannot fit TF-IDF on an empty list of complaints.")
        self.vectorizer.fit([normalize_for_tfidf(t) for t in texts])
        self.fitted = True
        return self

    def encode(self, texts):
        if not self.fitted:
            raise RuntimeError("Fit the TF-IDF model before encoding.")
        return self.vectorizer.transform([normalize_for_tfidf(t) for t in texts])

    @property
    def vocabulary_size(self):
        return len(self.vectorizer.vocabulary_) if self.fitted else 0
