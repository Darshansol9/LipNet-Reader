from read_alignment import read_align
from video_crop import read_video
import numpy as np
from numpy import array
import os
import math
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
import pickle



class Encodings:


    def __init__(self):

        self.model_vgg = VGG19(weights='imagenet', include_top=False)
        self.num_samples = 5
        self.dict_encodings = {}
        self.vocab_set = set()
        self.dict_text = {}
        self.max_len = -float('inf')
        self.ignore_video_files = set()

    def pad_images_to_maxlength(self):

        vgg_shape = self.dict_encodings[list(self.dict_encodings.keys())[0]].shape[1:]
        for k,v in self.dict_encodings.items():
            result = np.zeros((self.max_len,vgg_shape[0],vgg_shape[1],vgg_shape[2]))
            result[:v.shape[0], :v.shape[1], :v.shape[2], :v.shape[3]] = v
            self.dict_encodings[k] = result.flatten()

        path_to_save = '/scratch/vvt223/data/processed_data'
        
        with open(os.path.join(f'{path_to_save}','flatten_encodings.pickle'),'wb') as f:
            pickle.dump(self.dict_encodings, f, protocol=pickle.HIGHEST_PROTOCOL)

        
    def generateEncodingSamples(self,data,mode):


        error_file = open('/scratch/vvt223/data/processed_data/error_train_encoding_file.txt','w')
        count_samples = 0
        
        
        for paths in data:
          paths = paths.replace('\n','')
          read_align_path,video_align_path = paths.split('#')[0],paths.split('#')[1]
          folder,file = paths.split('/')[5],paths.split('/')[-1]
          fname,fext = os.path.splitext(file)
          
          if(fext == '.db'):
              continue
             
          alignments = read_align(read_align_path)
          mouth_video = read_video(video_align_path)
        
          word_count = 0
          first_occurence = False
          sentence = ['start']
          for start, stop, word in alignments:
              if word == 'sil' and not first_occurence:
                  first_occurence = True
                  continue

              try:              
                  if start < stop and stop < len(mouth_video):
                      img_frames = mouth_video[int(math.floor(start)):int(math.floor(stop))]
                      img_preprocess_frames = preprocess_input(img_frames)
                      img_features = self.model_vgg.predict(img_preprocess_frames)
                      self.dict_encodings[f'{folder}_{fname}_word{word_count}_{mode}'] = img_features
                      word_count +=1
                      if(mode == 'train' and word != 'sil'):
                          self.max_len = max(self.max_len,int(math.floor(stop))-int(math.floor(start)))
                          self.vocab_set.add(word)
                      if(word != 'sil'):
                          sentence.append(word)
                  else:
                    continue
              except Exception as e:
                  print('-'*40)
                  print(f'Cannot processed {folder}_{fname}_word{word_count}')
                  error_file.write(f'{folder}_{fname}_word{word_count}')
                  self.ignore_video_files.add(fname)
                  error_file.write('\n')
                  print('-'*40)
                  break
                

          sentence.append('end')
          if(not fname in self.ignore_video_files):
              self.dict_text[f'{folder}_{fname}_{mode}'] = sentence[::]

          print(f'Sample {count_samples} finished')
          count_samples +=1
          
          #if(count_samples == self.num_samples):
          #    print('Breaking')
          #    break

        error_file.close()

    def saveEncodings(self):

        path_to_save = '/scratch/vvt223/data/processed_data'
        
        with open(os.path.join(f'{path_to_save}','image_encodings.pickle'),'wb') as f:
            pickle.dump(self.dict_encodings, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(f'{path_to_save}','text_sentences.pickle'),'wb') as e:
            pickle.dump(self.dict_text, e, protocol=pickle.HIGHEST_PROTOCOL)


        vocabulary = ''
        vocabulary += "\n".join((str(v) for v in self.vocab_set))
        with open(os.path.join(f'{path_to_save}','corpus_words.txt'),'w') as f:
            f.write(vocabulary)
        f.close()

        ignore_videos = ''
        ignore_videos += "\n".join((str(v) for v in self.ignore_video_files))
        with open(os.path.join(f'{path_to_save}','ignore_videos_names.txt'),'w') as f:
            f.write(ignore_videos)
        f.close()

  

if __name__ == '__main__':
                  
    encod = Encodings()
    with open('/scratch/vvt223/data/X_train_full.txt','r') as f:
        data_train = f.readlines()
    f.close()

    with open('/scratch/vvt223/data/X_val_full.txt','r') as f:
        data_val = f.readlines()
    f.close()
    with open('/scratch/vvt223/data/X_test_full.txt','r') as f:
        data_test = f.readlines()
    f.close()

    print('------------Train Routine Started-------------------')
    encod.generateEncodingSamples(data_train,'train')
    print('------------Validation Routine Started-------------------')
    encod.generateEncodingSamples(data_val,'train')
    print('------------Test Routine Started-------------------')
    encod.generateEncodingSamples(data_test,'test')
    print('------------Encodings Saved-------------------')
    encod.saveEncodings()
    encod.pad_images_to_maxlength()
    print('------------Padded to max_length-------------------')
    print('Completed')
