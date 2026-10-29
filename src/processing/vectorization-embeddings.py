import pandas as pd
from gensim.models import Word2Vec

dataset = pd.read_csv("Data/Preprocessed/dataset_limpio.csv")

sentences = (dataset["texto_limpio"].str.split().tolist())

w2v_skipgram = Word2Vec(sentences,vector_size=150,window=5,min_count=5,sg=1,negative=5,epochs=10,workers=4,seed=67)
w2v_skipgram.save("Data/Final/word2vec_skipgram.model")

w2v_cbow = Word2Vec(sentences,vector_size=150,window=5,min_count=5,sg=0,negative=5,epochs=10,workers=4,seed=67)
w2v_cbow.save("Data/Final/word2vec_cbow.model")
