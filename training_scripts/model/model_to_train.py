
import tensorflow as tf

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))


from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Conv1D,MaxPooling1D,Bidirectional,LSTM, Embedding, Dense, Activation, Flatten, Reshape, concatenate, Dropout,add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras import Input, layers,optimizers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from data_pipeline import *
import matplotlib.pyplot as plt
import numpy as np
import os
from os import path
from tensorflow.keras.models import model_from_json
import json
from collections import defaultdict

class Test_Predictor:

    def __init__(self,model,max_length):

        ''' 
        Fetching all the encodings, text sentences and ignore files to remove corrupt videos files.
        '''

        lookup_path = '/scratch/vvt223/model_processed'
        data_path = '/scratch/vvt223/data'
        self.test_ignore_names = convert_to_set(read_text(os.path.join(lookup_path,'ignore_videos_names_darshan.txt')),'\n')
        self.test_names = convert_to_set(read_text(os.path.join(data_path,'X_test_full.txt')),'#')
        self.wordtoidx,self.idxtoword = read_pickle(lookup_path,'wordtoidx.pickle'),read_pickle(lookup_path,'idxtoword.pickle')
        self.max_length = max_length
        self.model = model

        print(self.test_ignore_names)
        print('Ignore names ',len(self.test_ignore_names))
        print('Total test names ',len(self.test_names))

        if not path.exists(os.path.join(lookup_path,'test_encodings_valid_darshan.pickle')):

            print('-----------------Processed Test encodings Getting Saved OTP ----------------------')
            self.test_encodings = read_pickle(lookup_path,'image_encodings_darshan.pickle')
            self.text_sentence = read_pickle(lookup_path,'text_sentences_darshan.pickle')
            self.flattened_test_encodings = pad_to_max_length(self.test_encodings)
            self.valid_test_encodings = self.flattened_test_encodings
            save_pickle(self.valid_test_encodings,lookup_path,'test_encodings_valid_darshan.pickle')

        else:
            print('-------------Readups for test prediction--------------------------')
            self.valid_test_encodings = read_pickle(lookup_path,'test_encodings_valid_darshan.pickle')

        self.seq_test = read_pickle(lookup_path,'test_valid_sentences_darshan.pickle')
        self.dict_video_word_count = read_pickle(lookup_path,'video_word_counts_darshan.pickle')
        self.map_name_to_text_test = read_pickle(lookup_path,'names_to_filename_test_darshan.pickle')

        #print('Test encodings length ',len(self.valid_test_encodings))
        #print('Text sentence ',len(self.seq_test))
        #print('Word count dictionary ',len(self.dict_video_word_count))
        #print('Mapping length ',len(self.map_name_to_text_test))

    def predict(self,photo):

        ''' 
        Takes in the entire video in sequence and given a start token model predicts the next word
        '''
  
        in_text = 'start'
        j = 0
        for i in range(self.max_length):
            sequence = [self.wordtoidx[w] for w in in_text.split() if w in self.wordtoidx]
            sequence = pad_sequences([sequence], maxlen=self.max_length)

            if(j == len(photo)):
              break
            
            photo[j] = photo[j].reshape(1,-1)
            sequence  = sequence.reshape(1,-1)

            yhat = self.model.predict([np.array(photo[j]),np.array(sequence)], verbose=0)
            yhat = np.argmax(yhat)
            word = self.idxtoword[yhat]
            in_text += ' ' + word

            if word == 'end':
                break
            j+=1
              
        final = in_text.split()
        final = final[1:-1]
        final = ' '.join(final)

        return final


    def calWER(self,actual_sentence,pred_sentence):

        '''
        given predicted sentence and actual sentence compute index-based incorrect words

        '''

        pred_tok = pred_sentence.split(' ')
        correct = 0

        n = len(pred_tok)
        m = len(actual_sentence)

        for i in range(min(n,m)):
            if(pred_tok[i] != actual_sentence[i]):
                correct +=1

        return correct / max(n,m)




    def eval(self):

        incorrect_dict = collections.defaultdict(dict)

        def incorrect_predicted_words(sent1,sent2):

            n,m = len(sent1),len(sent2)
            for i in range(min(n,m)):
                if(sent1[i] != sent2[i]):
                    incorrect_dict[sent1[i]][sent2[i]] += 1
    
        print('-------------------------------Evaluating the model for prediction ------------------------------------------------')
        total = 0
        for k,v in self.dict_video_word_count.items():
            get_encodings = []
            key_name_tokens = self.map_name_to_text_test[k].split('_')
            print(key_name_tokens)
            for i in range(v):
                form_key = key_name_tokens[0]+'_'+key_name_tokens[1]+'_'+f'word{i}'+'_'+key_name_tokens[2]
                get_encodings.append(self.valid_test_encodings[form_key])

            pred_sentence = self.predict(get_encodings)
            actual_sentence = self.seq_test[self.map_name_to_text_test[k]]

            incorrect_predicted_words(actual_sentence,pred_sentence.split(' '))
            print(f'Video {key_name_tokens[1]} prediction pred_sentence ',pred_sentence)
            print(f'Video {key_name_tokens[1]} actual sentence ',' '.join(actual_sentence))
            print('-'*40)

            total += self.calWER(actual_sentence,pred_sentence)
        print('WER Error: ',total)

        print('------------------------------Printing number of incorrect words occured---------------------------------------------') 
        
        for k,v in incorrect_dict.items():
            temp_dict = dict(sorted(v.items(),key=lambda x: x[1],reverse=True))
            print('Correct Word {k} incorrectly predicted with : ')
            count = 0
            for k1,v1 in temp_dict.items():
                print(f'{k1} number of times {v1}')
                count +=1

                if(count == 5):
                    break
        


class CNN_LSTM:

    def helper_for_read(self):
        
        lookup_path = '/scratch/vvt223/model_processed'
        pickle_name = 'flattened_encodings_vishnu_expand.pickle'
        pickle_name1 = 'text_sentences_vishnu.pickle'
        print('-------------------------Reading data encodings pickle files ----------------------------')
        self.image_encodings = read_pickle(lookup_path,pickle_name)
        self.text_sentences = read_pickle(lookup_path,pickle_name1)
        print('-------------------------Done Reading pickle files---------------------------------------')
        self.vocab = convert_to_set(read_text(lookup_path,'corpus_words.txt'),'\n')
        self.videofiles_ignore = convert_to_set(read_text(lookup_path,'ignore_videos_names.txt'),'\n')
        self.val_file_name = convert_to_set(read_text(lookup_path,'X_val_full.txt'),'\n')
        self.train_file_name = convert_to_set(read_text(lookup_path,'X_train_full.txt'),'\n')
        print('Done Reading all data....')
    
    def helper_for_compute(self):

        self.wordtoidx,self.idxtoword = word_to_index(self.vocab)
        save_pickle(self.wordtoidx,'/scratch/vvt223/model_processed','wordtoidx.pickle')
        save_pickle(self.idxtoword,'/scratch/vvt223/model_processed','idxtoword.pickle')

        self.max_length = to_max_length(self.text_sentences.values())
        self.vocab_size = len(self.vocab) + 1

        #generating the embedding matrix
        self.embedding_matrix = generateEmbeddings(self.text_sentences.values(),
            self.vocab_size,
            self.wordtoidx,
            self.max_length,
            self.embedding_dim
            )

        #making train val split with cleanup on ignore video files.
        self.train_encodings,self.seq_train,self.val_encodings,self.seq_val,self.dict_text_to_names = train_val_split(
            self.image_encodings,
            self.text_sentences,
            self.videofiles_ignore,
            self.val_file_name,
            self.train_file_name
            )

        print('Done computing all the values....')

    def __init__(self):

        self.embedding_dim = 100
        self.helper_for_read()
        self.helper_for_compute()

        #initializer
        self.input_size = self.image_encodings[list(self.image_encodings.keys())[0]].shape
        self.dense_size = 128
        self.lstm_cell = 128
        self.dropout = 0.1
        self.epochs = 500
        self.num_videos_per_batch = 256
        self.steps_per_epoch = len(self.text_sentences)//self.num_videos_per_batch
        self.validation_steps = 100
        self.model = None

    def build(self):

        model1 = Sequential()
        model1.add(Conv1D(filters=32, kernel_size=9, activation='relu',input_shape = self.input_size))
        model1.add(MaxPooling1D(pool_size=7))
        model1.add(Dropout(self.dropout))
        model1.add(Conv1D(filters=16, kernel_size=9, activation='relu'))
        model1.add(MaxPooling1D(pool_size=7))
        model1.add(Dropout(self.dropout))
        model1.add(Flatten())
        model1.add(Dropout(self.dropout+0.2))
        model1.add(Dense(512,activation='relu',kernel_regularizer=tf.keras.regularizers.L1(0.01),
                activity_regularizer=tf.keras.regularizers.L2(0.01)))
        model1.add(Dropout(self.dropout+0.2))
        model1.add(Dense(256,activation='relu',kernel_regularizer=tf.keras.regularizers.L1(0.01),
                activity_regularizer=tf.keras.regularizers.L2(0.01)))

        model2  = Sequential()
        model2.add(Input(shape=(self.max_length,)))
        model2.add(Embedding(self.vocab_size, self.embedding_dim, mask_zero=True))
        model2.add(Dropout(self.dropout+0.2))
        model2.add(Bidirectional(LSTM(self.lstm_cell)))

        decoder1= add([model1.layers[-1].output, model2.layers[-1].output])
        decoder2 = Dense(self.dense_size, activation='relu')(decoder1)

        outputs = Dense(self.vocab_size, activation='softmax')(decoder2)

        model = Model(inputs=[model1.inputs,model2.inputs], outputs=outputs)
        self.model = model
        print('---------------------------------Model Summary--------------------------\n',self.model.summary())

        #initializing embedding layer matrix
        self.model.layers[11].set_weights([self.embedding_matrix])
        self.model.layers[11].trainable = False

        print('----------------Building the model-------------------------')
        adam = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,name='Adam')
        self.model.compile(loss='categorical_crossentropy', optimizer=adam,metrics=['accuracy'])
        

        print('--------------------Model Compiled-------------------------')

        model_json = self.model.to_json()
        with open("/scratch/vvt223/model_processed/model_weights/model.json", "w") as json_file:
            json_file.write(model_json)

        print('------------------Model Architecture Saved--------------------')


    def data_generator(self,seq_, dict_encoding, wordtoidx, max_length,vocab_size,num_images_per_batch):
        X1, X2, y = list(), list(), list()
        n=0

        while True:

            for key,tokenized_sentence in seq_.items():
                n+=1
                window_size = 2
                fol_name,video_name,mode = key.split('_')[0],key.split('_')[1],key.split('_')[-1]
                seq = [wordtoidx[word] for word in tokenized_sentence]
                for i in range(1,len(seq)-1):
                    
                    #this is crux where we generate the sequence of window size 2
                    img = dict_encoding[f'{fol_name}_{video_name}_word{i-1}_{mode}']
                    in_seq, out_seq = seq[i-window_size:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length,value=0)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size+1)[0]
                    
                    #taking curr image and previous word to predict next word
                    X1.append(img)
                    X2.append(in_seq)
                    y.append(out_seq)

                if n==num_images_per_batch:
                    #print('Batch fullfilled')
                    yield ((np.array(X1), np.array(X2)), np.array(y))
                    X1, X2, y = list(), list(), list()
                    n=0
            

    def train(self):

        """Implementing datagenerator, training model and plotting losses for end result"""

        callback_es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                        min_delta=0, 
                                        patience=10, 
                                        verbose=1, 
                                        mode='auto', 
                                        baseline=None, 
                                        restore_best_weights=True)

        #checkpoint_filepath = '/content/drive/MyDrive/' + 'weights_best.hdf5'
        checkpoint_filepath = '/scratch/vvt223/model_processed/model_weights/weights_best.hdf5'
        best_weight_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                            filepath=checkpoint_filepath,
                            save_weights_only=True,
                            monitor='val_loss',
                            mode='auto',
                            save_best_only=True,
                            verbose=1)

        train_generator = self.data_generator(self.seq_train, self.train_encodings, self.wordtoidx, self.max_length,len(self.vocab),self.num_videos_per_batch)
        val_generator = self.data_generator(self.seq_val, self.val_encodings, self.wordtoidx, self.max_length,len(self.vocab),self.num_videos_per_batch)
        history = self.model.fit(train_generator,
              validation_data=val_generator,
              validation_steps=self.validation_steps,
              epochs=self.epochs,
              steps_per_epoch=self.steps_per_epoch,
              callbacks=[best_weight_checkpoint,callback_es]
            )

        #Plotting loss
        self.lossPlot(history.history,'loss')
        self.lossPlot(history.history,'accuracy')
        print('-------------Plots Saved--------------------')

    def lossPlot(self,history,name):
		
        EPOCHS = self.epochs
        path_to_save = '/scratch/vvt223/model_processed/plots'
        plt.plot(range(1,EPOCHS+1),history[f'val_{name}'],'-',linewidth=3,label=f'Val {name}')
        plt.plot(range(1,EPOCHS+1),history['f{name}'],'-',linewidth=3,label=f'Train {name}')
        plt.xlabel('epoch')
        plt.ylabel('{name}')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(path_to_save,f'{name}plot.png'))


    

if __name__ == '__main__':

    model_saved_path = '/scratch/vvt223/model_processed/model_weights/weights_best.hdf5'

    if not path.exists(model_saved_path):
        obj_mod = CNN_LSTM()
        print('-------Building the model -----')
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            obj_mod.build()
        print('-------Training has started ----')
        obj_mod.train()
        print('-------Training completed-------')

    
    print(f'Loading Best Model for Testing in process..')

    json_file = open('/scratch/vvt223/model_processed/model_weights/model.json', 'r')
    loaded_model = json_file.read()
    json_file.close()

    loaded_model = model_from_json(loaded_model)
    loaded_model.load_weights(model_saved_path)

    loaded_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

    print('-----Test Routine Started------------')
    
    predictor = Test_Predictor(loaded_model,10)
    predictor.eval()

