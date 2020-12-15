from pickle5 import pickle
import os
from gensim.models import Word2Vec
import numpy as np
import nltk
nltk.download('punkt')

def save_text(data,path_to_save,name):

  str_data = ''
  str_data += "\n".join((str(v) for v in data))
  with open(os.path.join(f'{path_to_save}',name),'w') as f:
      f.write(str_data)
      f.close()

def save_pickle(data,path_to_save,name):

  with open(os.path.join(path_to_save,name),'wb') as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def make_test_for_prediction(dict_encoding,text_sequence,ignore_names,test_names):
  
  print('Inside Make Test for Prediction')

  dict_text_to_name = {k.split('_')[1]:k for k,v in text_sequence.items()}
  dict_encoding_test = {}
  seq_test = {}
  dict_video_word_count = {}

  for k,v in dict_encoding.items():
    fname = k.split('_')[1]
    if(fname not in ignore_names):
      if(fname in test_names):

        if(fname in dict_video_word_count):
          dict_video_word_count[fname] +=1
        else:
          dict_video_word_count[fname] = 1

        dict_encoding_test[k] = v
        seq_test[dict_text_to_name[fname]] = text_sequence[dict_text_to_name[fname]]

  return dict_encoding_test,seq_test,dict_text_to_name,dict_video_word_count

def pad_to_max_length(dict_encodings):

  vgg_shape = dict_encodings[list(dict_encodings.keys())[0]].shape[1:]
  for k,v in dict_encodings.items():
      result = np.zeros((33,vgg_shape[0],vgg_shape[1],vgg_shape[2]))
      if(v.shape[0] >= 33):
        result[:33, :v.shape[1], :v.shape[2], :v.shape[3]] = v[:33,:,:,:]
      else:
        result[:v.shape[0],:v.shape[1],:v.shape[2],:v.shape[3]] = v
       
      dict_encodings[k] = np.expand_dims(result.flatten(),axis=1)
  
  return dict_encodings

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



def train_val_split(dict_encoding,text_sequence,ignore_names,val_names,train_names):

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

  vocab.add('start')
  vocab.add('end')

  wordtoidx = {}
  idxtoword = {}
  
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
