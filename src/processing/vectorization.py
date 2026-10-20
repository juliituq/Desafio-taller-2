from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer

NEGATIONS = ["no", "not", "nor", "never","none", "nobody", "nothing", "neither", "nowhere", "hardly", "scarcely", "barely", "cannot"]
STOPWORDS = list(set(stopwords.words('english')) - set(NEGATIONS))

dataset = pd.read_csv("Data/Preprocessed/dataset_limpio.csv")

def bow_vectorizer(documents, n_gram):
    bow = CountVectorizer(ngram_range=n_gram, stop_words=STOPWORDS, analyzer='word')
    result = bow.fit_transform(documents)
    return pd.DataFrame(result.toarray(), columns=bow.get_feature_names_out())

def tfidf_vectorizer(documents, n_gram):
    tfidf = TfidfVectorizer(ngram_range=n_gram, stop_words=STOPWORDS)
    result = tfidf.fit_transform(documents)
    return pd.DataFrame(result.toarray(), columns=tfidf.get_feature_names_out())

bow_result = bow_vectorizer(dataset["texto_limpio"], n_gram=(1, 1))
bow_result.to_csv("Data/Final/bow_result_unigram.csv", index=False)

bow_result_bigram = bow_vectorizer(dataset["texto_limpio"], n_gram=(1, 2))
bow_result_bigram.to_csv("Data/Final/bow_result_unigram_bigram.csv", index=False)
tfidf_result = tfidf_vectorizer(dataset["texto_limpio"], n_gram=(1, 1))
tfidf_result.to_csv("Data/Final/tfidf_result_unigram.csv", index=False)

print(bow_result.head())
print(tfidf_result.head())