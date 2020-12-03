from model2_datapipeline import read_data,compute_vocab,to_max_length
from gensim.models import Word2Vec
import numpy as np
import nltk
nltk.download('punkt')


def generateEmbeddings():

	sequence_dict,vocab_list = read_data('/content/data_processed','word_embedding')
	sequence = list(sequence_dict.values())
	max_length = to_max_length(sequence)
	vocab_size = len(compute_vocab(vocab_list)) + 1
	w2v = Word2Vec(sequence, size=100, window=5, min_count=1, workers=4)

	#generate embeddings
	embedding_dim = 100
	embedding_matrix = np.zeros((vocab_size, embedding_dim))
	for word, i in wordtoidx.items():
	    embedding_vector = w2v.wv[word]
	    embedding_matrix[i] = embedding_vector


	 return embedding_matrix
