from pickle5 import pickle
import os
from gensim.models import Word2Vec
import numpy as np
import nltk
nltk.download('punkt')



def generateEmbeddings(sequence,vocab_size,wordtoidx,max_length,embedding_dim):

  w2v = Word2Vec(sequence, size=embedding_dim, window=5, min_count=1, workers=4)
  embedding_matrix = np.zeros((vocab_size, embedding_dim))
  for word, i in wordtoidx.items():
      embedding_vector = w2v.wv[word]
      embedding_matrix[i] = embedding_vector

  return embedding_matrix


def convert_to_set(data,delimiter):

  unique_names = set()
  for row in data:
    row = row.split(delimiter)[0]
    fname,fext = os.path.splitext(row)
    unique_names.add(fname)

  return unique_names


  
def train_val_split(dict_encoding,text_sequence,ignore_file_name,val_file_name,train_file_name):

  ignore_names = convert_to_set(ignore_file_name,'#')
  val_names = convert_to_set(val_file_name,'#')
  train_names = convert_to_set(train_file_name,'#')
  dict_text_to_name = {k.split('_')[1]:k for k,v in text_sequence.items()}
  dict_encoding_train = {}
  dict_encoding_val = {}
  seq_train = {}
  seq_val = {}

  for k,v in dict_encoding.items():
    fname = k.split('_')[1]
    if(fname not in ignore_names):
      if(fname in train_names):
        dict_encoding_train[k] = v
        seq_train[dict_text_to_name[fname]] = text_sequence[dict_text_to_name[fname]]
      else:
        dict_encoding_val[k] = v
        seq_val[dict_text_to_name[fname]] = text_sequence[dict_text_to_name[fname]]

  return dict_encoding_train,seq_train,dict_encoding_val,seq_val,dict_text_to_name


def read_pickle(dir_to_look,name):

  f = open(os.path.join(dir_to_look,name),'rb')
  return pickle.load(f)


def read_text(dir_to_look,name=None):

  if(name is not None):
    with open(os.path.join(dir_to_look,name),'r') as f:
      data = f.readlines()

  else:
    with open(dir_to_look,'r') as f:
      data = f.readlines()

  return data


def compute_vocab(vocab_lst):
    
  for w in vocab_lst.split('\n'):
      if(w == ''):
          continue
        
      else:
          vocab.add(w)

  vocab.add('start')
  vocab.add('end')
    
  return vocab

def word_to_index(vocab=set()):

  start_seq = 0
  end_seq = 0

  for i,val in enumerate(vocab):

    if(val == 'start'):
      start_seq = i
      continue
      
    if(val == 'end'):
      end_seq = i
      continue

    wordtoidx[val] = i+1
    idxtoword[i+1] = val

  wordtoidx['start'] = start_seq
  idxtoword[start_seq] = 'start'
  wordtoidx['end'] = end_seq
  idxtoword[end_seq] = 'end'
  
  return wordtoidx,idxtoword

def to_max_length(sequence):
  
  return max(len(d) for d in sequence)

'''
dict_encoding,sequence,vocab_lst = read_data('/content/data_processed')
compute_vocab(vocab_lst)
word_to_index(vocab)
max_length = to_max_length(sequence)
'''
