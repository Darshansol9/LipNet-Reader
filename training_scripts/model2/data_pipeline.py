import pickle
import os

def read_data(dir_to_look,call_func):

  
  f = open(os.path.join(dir_to_look,'dict_text.pickle'),'rb')
      sequence = pickle.load(f)

  with open(os.path.join(dir_to_look,'corpus_text.txt'),'r') as f:
    vocab_lst = f.readlines()

  if(call_func == 'word_embedding'):
      return sequence,vocab_lst

  else:
    e = open(os.path.join(dir_to_look,'dict_encodings.pickle'),'rb')
    dict_encoding = pickle.load(e)

    return dict_encoding,sequence,vocab_lst


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
