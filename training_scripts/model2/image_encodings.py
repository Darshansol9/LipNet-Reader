##%tensorflow_version 2.x
##import tensorflow as tf
##device_name = tf.test.gpu_device_name()
##if device_name != '/device:GPU:0':
##  raise SystemError('GPU device not found')
##print('Found GPU at: {}'.format(device_name))

from read_alignments import read_align
from video_crop import read_video
import numpy as np
from numpy import array
import os
import math
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
import pickle



class Encodings:


    def __init__(self):

        self.model_vgg = VGG19(weights='imagenet', include_top=False)
        self.num_samples = 1
        self.dict_encodings = {}
        self.vocab_set = set()
        self.dict_text = {}


    def generateEncodings(self,dir_path,mode):

      for folder in os.listdir(dir_path):

        s_path = os.path.join(dir_path,folder)
        #print(s_path)

        count_samples = 0

        for files in os.listdir(s_path):

          fname,fext = os.path.splitext(files)
          alignments = read_align(os.path.join(os.path.join(s_path,'align'),fname+'.align'))
          mouth_video = read_video(os.path.join(s_path,fname+'.mpg'))
        
          word_count = 0
          sentence = ['start']
          for start, stop, word in alignments:
              if word == 'sil' or word == 'sp':
                continue
                                          
              if start < stop and stop < len(mouth_video):
                  img_frames = mouth_video[int(math.floor(start)):int(math.floor(stop))]
                  img_preprocess_frames = preprocess_input(img_frames)
                  img_features = self.model_vgg.predict(img_preprocess_frames)
                  self.dict_encodings[f'{folder}_{fname}_word{word_count}_{mode}'] = img_features
                  word_count +=1
                  if(mode == 'train'):
                    self.vocab_set.add(word)
              else:
                continue

          sentence.append('end')
          self.dict_text[f'{folder}_{fname}_{mode}'] = sentence[::]
          
          if(count_samples == self.num_samples):
            break

          print(f'Sample {count_samples} finished for speaker {folder}')
          count_samples +=1


    def saveEncodings(self):

        path_to_save = '/content/data_processed/'
        
        with open(os.path.join(f'{path_to_save}','image_encodings.pickle'),'wb') as f:
            pickle.dump(self.dict_encodings, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(f'{path_to_save}','text_encodings.pickle'),'wb') as e:
            pickle.dump(self.dict_text, e, protocol=pickle.HIGHEST_PROTOCOL)

        vocabulary = ''
        vocabulary += "\n".join((str(v) for v in self.vocab_set))
        with open(os.path.join(f'{path_to_save}','corpus_words.txt'),'w') as f:
            f.write(vocabulary)
        f.close()

  

if __name__ == '__main__':
                  
    encod = Encodings()
    encod.generateEncodings('/content/seen_speakers/',mode='train')
    encod.generateEncodings('/content/unseen_speakers/',mode='test')
    encod.saveEncodings()
