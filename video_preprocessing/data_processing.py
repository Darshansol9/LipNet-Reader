from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import os
import numpy as np
import cv2
import dlib
import math
from read_alignment import read_align
from video_crop import read_video


def data_generator(BASE_PATH,X,y,testing=False):

  if(not testing):
    max_word_len = -float('inf')
    max_len = -float('inf')

  #BASE_PATH = r'/scratch/vvt223/data/'

  j = 1
  while j <= 34:

    if(not testing):
      if(j == 1 or j == 2 or j = 20 or j == 22):
        j+=1
        continue

    dir_path = os.path.join(BASE_PATH,f's{j}/video')

    for files in os.listdir(dir_path):
      
      count +=1
      fname,fext = os.path.splitext(files)

      alignments = read_align(os.path.join(os.path.join(BASE_PATH,f's{j}/align'),fname+'.align'))
      mouth_video = read_video(os.path.join(dir_path,fname+'.mpg'))
      
      for start, stop, word in alignments:
          if word == 'sil' or word == 'sp':
              continue
            
          if(testing):
            
                        
          if start < stop and stop < len(mouth_video):
              X.append(mouth_video[int(math.floor(start)):int(math.floor(stop))])
              y.append(word)
          else:
              continue

          if(not testing):
            max_word_len = max(max_word_len, len(word))
            max_len = max(max_len, int(math.floor(stop)) - int(math.floor(start)))

    j+=1

  if(not testing):
    return X,y,max_word_len,max_len

  else:
    return X,y


def main():

  oh = OneHotEncoder()
  le = LabelEncoder()

  BASE_PATH = r'/scratch/vvt223/data/'

  X_test,y_test = data_generator(os.path.join(BASE_PATH,'unseen_speakers'),[],[],True)
  X_train,y_train,max_word_len,max_len = data_generator(os.path.join(BASE_PATH,'seen_speakers'),[],[],False)

  for i in range(len(X_train)):
    result = np.zeros((max_len, 45, 70, 3))
    result[:X_train[i].shape[0], :X_train[i].shape[1], :X_train[i].shape[2], :X_train[i].shape[3]] = X_train[i]
    X_train[i] = result


  for j in range(len(X_test)):

    #capturing the case where max_len from train is less then X_test word frame utterred
    
    if(X_test[j].shape[0] > max_len):

      diff = X_test[j].shape[0] - max_len
      start  = diff // 2
      X_test[j] = X_test[j][start:max_len+start,:,:,:]

    result = np.zeros((max_len, 45, 70, 3))
    result[:X_test[j].shape[0], :X_test[j].shape[1], :X_test[j].shape[2], :X_test[j].shape[3]] = X_test[j]
    X_test[i] = result

  #Fit and transform for train
  y_train = le.fit_transform(y_train)
  y_train = oh.fit_transform(y_train.reshape(-1,1)).todense()
  x_train = np.stack(X_train,axis=0)

  #Transforming the output of test to avoid bias in the model
  y_test = le.transform(y_test)
  y_test = oh.transform(y_test.reshape(-1,1)).todense()
  x_test = np.stack(X_test,axis=0)

  print('Saving processed data ...\n')

  path_to_save = r'/scratch/vvt223/processed_data/'

  np.savez_compressed(os.path.join(path_to_save,'X_train'), x=x_train)
  np.savez_compressed(os.path.join(path_to_save,'y_train'), y=y_train)
  np.savez_compressed(os.path.join(path_to_save,'X_test'), x=x_test)
  np.savez_compressed(os.path.join(path_to_save,'y_test'), y=y_test)

  print(f'Data saved at {path_to_save}')



if __name__ == '__main__':

  main()
