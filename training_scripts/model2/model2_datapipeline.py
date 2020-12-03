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

def data_generator(sequence, images, wordtoix, max_length,vocab_size,num_images_per_batch):
    X1, X2, y = list(), list(), list()
    n=0

    while True:

      for idx,pack in enumerate(sequence):

        key = pack[0]
        tokenized_sentence = pack[1]

        fol_name,video_name,mode = key.split('_')

        n+=1

        seq = [wordtoidx[word] for word in tokenized_sentence]
        #print(seq)
        for i in range(1,len(seq)):

          img = dict_encoding[f'{fol_name}_{video_name}_w{(i-1)%7}_{mode}']
          in_seq, out_seq = seq[:i], seq[i]
          
          #padding the input and output sequence
          in_seq = pad_sequences([in_seq], maxlen=max_length,value=0)[0]
          out_seq = to_categorical([out_seq], num_classes=vocab_size+1)[0]

          #taking curr image and previous word to predict next word
          X1.append(img)
          X2.append(in_seq)
          y.append(out_seq)

        #taking 256 batches at a time
        #print(n)
        if n==num_images_per_batch:
          #print('Batch fullfilled')
          yield ((np.array(X1), np.array(X2)), np.array(y))
          X1, X2, y = list(), list(), list()
          n=0
  
      #print('Executing break')
      #break
'''      
dict_encoding,sequence,vocab_lst = read_data('/content/data_processed')
compute_vocab(vocab_lst)
word_to_index(vocab)
max_length = to_max_length(sequence)
generator = data_generator(sequence,dict_encoding, wordtoidx,max_length,len(vocab),5) #num_of_videos_per_batch
'''
