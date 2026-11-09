from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import save_npz
import json

NEGATIONS = ["no", "not", "nor", "never","none", "nobody", "nothing", "neither", "nowhere", "hardly", "scarcely", "barely", "cannot"]
STOPWORDS = list(set(stopwords.words('english')) - set(NEGATIONS))

train = pd.read_csv("Data/Splits/train.csv")
validacion = pd.read_csv("Data/Splits/validacion.csv")
test = pd.read_csv("Data/Splits/test.csv")

def bow_vectorizer(documents, documents_validacion, documents_test, n_gram):
    bow = CountVectorizer(ngram_range=n_gram, stop_words=STOPWORDS, analyzer='word')
    result = bow.fit_transform(documents)
    result_validacion = bow.transform(documents_validacion)
    result_test = bow.transform(documents_test)
    return result, result_validacion, result_test, bow.get_feature_names_out()

def tfidf_vectorizer(documents, documents_validacion, documents_test, n_gram):
    tfidf = TfidfVectorizer(ngram_range=n_gram, stop_words=STOPWORDS)
    result = tfidf.fit_transform(documents)
    result_validacion = tfidf.transform(documents_validacion)
    result_test = tfidf.transform(documents_test)
    return result, result_validacion, result_test, tfidf.get_feature_names_out()

def save_features(features, filename):
    with open(f"Data/Final/{filename}.json", "w", encoding="utf-8") as f:
        json.dump(features.tolist(), f, ensure_ascii=False)

bow_result, bow_validacion, bow_test, bow_features = bow_vectorizer(
    train["texto_limpio"].fillna(""),
    validacion["texto_limpio"].fillna(""),
    test["texto_limpio"].fillna(""),
    n_gram=(1, 1)
)
save_npz("Data/Final/bow_result_unigram_train.npz", bow_result)
save_npz("Data/Final/bow_result_unigram_validacion.npz", bow_validacion)
save_npz("Data/Final/bow_result_unigram_test.npz", bow_test)
save_features(bow_features, "bow_features_unigram")

bow_result_bigram, bow_validacion_bigram, bow_test_bigram, bow_features_bigram = bow_vectorizer(
    train["texto_limpio"].fillna(""),
    validacion["texto_limpio"].fillna(""),
    test["texto_limpio"].fillna(""),
    n_gram=(1, 2)
)
save_npz("Data/Final/bow_result_unigram_bigram_train.npz", bow_result_bigram)
save_npz("Data/Final/bow_result_unigram_bigram_validacion.npz", bow_validacion_bigram)
save_npz("Data/Final/bow_result_unigram_bigram_test.npz", bow_test_bigram)
save_features(bow_features_bigram, "bow_features_unigram_bigram")

tfidf_result, tfidf_validacion, tfidf_test, tfidf_features = tfidf_vectorizer(
    train["texto_limpio"].fillna(""),
    validacion["texto_limpio"].fillna(""),
    test["texto_limpio"].fillna(""),
    n_gram=(1, 1)
)
save_npz("Data/Final/tfidf_result_unigram_train.npz", tfidf_result)
save_npz("Data/Final/tfidf_result_unigram_validacion.npz", tfidf_validacion)
save_npz("Data/Final/tfidf_result_unigram_test.npz", tfidf_test)
save_features(tfidf_features, "tfidf_features_unigram")

tfidf_result_bigram, tfidf_validacion_bigram, tfidf_test_bigram, tfidf_features_bigram = tfidf_vectorizer(
    train["texto_limpio"].fillna(""),
    validacion["texto_limpio"].fillna(""),
    test["texto_limpio"].fillna(""),
    n_gram=(1, 2)
)
save_npz("Data/Final/tfidf_result_unigram_bigram_train.npz", tfidf_result_bigram)
save_npz("Data/Final/tfidf_result_unigram_bigram_validacion.npz", tfidf_validacion_bigram)
save_npz("Data/Final/tfidf_result_unigram_bigram_test.npz", tfidf_test_bigram)
save_features(tfidf_features_bigram, "tfidf_features_unigram_bigram")