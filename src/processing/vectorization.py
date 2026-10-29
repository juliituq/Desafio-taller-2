from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import save_npz
import json

NEGATIONS = ["no", "not", "nor", "never","none", "nobody", "nothing", "neither", "nowhere", "hardly", "scarcely", "barely", "cannot"]
STOPWORDS = list(set(stopwords.words('english')) - set(NEGATIONS))

dataset = pd.read_csv("Data/Preprocessed/dataset_limpio.csv")

def bow_vectorizer(documents, n_gram):
    bow = CountVectorizer(ngram_range=n_gram, stop_words=STOPWORDS, analyzer='word')
    result = bow.fit_transform(documents)
    return result, bow.get_feature_names_out()

def tfidf_vectorizer(documents, n_gram):
    tfidf = TfidfVectorizer(ngram_range=n_gram, stop_words=STOPWORDS)
    result = tfidf.fit_transform(documents)
    return result, tfidf.get_feature_names_out()

def save_features(features, filename):
    with open(f"Data/Final/{filename}.json", "w", encoding="utf-8") as f:
        json.dump(features.tolist(), f, ensure_ascii=False)

bow_result, bow_features = bow_vectorizer(dataset["texto_limpio"], n_gram=(1, 1))
save_npz("Data/Final/bow_result_unigram.npz", bow_result)
save_features(bow_features, "bow_features_unigram")

bow_result_bigram, bow_features_bigram = bow_vectorizer(dataset["texto_limpio"], n_gram=(1, 2))
save_npz("Data/Final/bow_result_unigram_bigram.npz",bow_result_bigram)
save_features(bow_features_bigram, "bow_features_unigram_bigram")

tfidf_result, tfidf_features = tfidf_vectorizer(dataset["texto_limpio"], n_gram=(1, 1))
save_npz("Data/Final/tfidf_result_unigram.npz", tfidf_result)
save_features(tfidf_features, "tfidf_features_unigram")

tfidf_result_bigram, tfidf_features_bigram = tfidf_vectorizer(dataset["texto_limpio"], n_gram=(1, 2))
save_npz("Data/Final/tfidf_result_unigram_bigram.npz",tfidf_result_bigram)
save_features(tfidf_features_bigram, "tfidf_features_unigram_bigram")
