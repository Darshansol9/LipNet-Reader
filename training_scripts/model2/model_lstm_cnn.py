
import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

from tensorflow.keras.layers.core import Dense, Activation, Dropout, Flatten
from tensorflow.keras.layers import LSTM, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras import Input, layers
from tensorflow.keras import optimizers
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from word_embed import generateEmbeddings
from model2.datapipeline import read_data,compute_vocab,to_max_length


class CNN_LSTM:


	def helper_for_read(self):

		lookup_path = '/scratch/vvt223/processed_data'
		self.image_encodings = read_pickle(lookup_path,'flatten_encodings.pickle')
		self.text_sentences = read_pickle(lookup_path,'text_sentences.pickle')
		self.vocab = convert_to_set(read_text(lookup_path,'corpus_words.txt'),'\n')
		self.videofiles_ignore = convert_to_set(read_text(lookup_path,'ignore_video_files.txt'),'\n')
		self.val_file_name = convert_to_set(read_text(lookup_path,'X_val_3000.txt'),'#')
		self.train_file_name = convert_to_set(read_text(lookup_path,'X_train_10000.txt'),'#')
		self.test_file_name = convert_to_set(read_text(lookup_path,'X_test_2000.txt'),'#')

	def helper_for_compute(self):


		self.vocab.add('start')
		self.vocab.add('end')
		self.wordtoidx,self.idxtoword = word_to_index(self.vocab)
		self.max_length = to_max_length(self.text_sentences.values())
		self.vocab_size = len(self.vocab) + 1

		self.embedding_matrix = generateEmbeddings(self.text_sentences.values(),
			self.vocab_size,
			self.wordtoidx,
			self.max_length,
			self.embedding_dim
			)
		self.train_encodings,self.seq_train,self,val_encodings,self.seq_val,self.dict_text_to_names = train_val_split(
			self.image_encodings,
			self.text_sentences,
			self.videofiles_ignore,
			self.val_file_name,
			self.train_file_name
			)

	def __init__(self):

		self.embedding_dim = 100
		self.helper_for_read()
		self.helper_for_compute()

		#initializer
		self.input_size = self.image_encodings[list(self.image_encodings.keys())[0]].shape
		self.dense_size = 2048
		self.lstm_cell = 256
		self.dropout = 0.2
		self.epochs = 30
		self.num_videos_per_batch = 256
		self.steps = len(self.text_sentences)//self.num_videos_per_batch
		self.validation_steps = 20
		self.model = None

	def build(self):

		inputs1 = Input(shape=(self.input_size,))
		fe1 = Dropout(self.dropout)(inputs1)
		fe2 = Dense(self.dense_size, activation='relu')(fe1)
		fe3 = Dropout(self.dropout)(fe2)
		fe4 = Dense(self.dense_size//2,activation='relu')(fe3)
		inputs2 = Input(shape=(self.max_length,))
		se1 = Embedding(self.vocab_size, self.embedding_dim, mask_zero=True)(inputs2)
		se2 = Dropout(self.dropout)(se1)
		se3 = LSTM(self.lstm_cell)(se2)
		decoder1 = add([fe4, se3])
		decoder2 = Dense(self.lstm_cell, activation='relu')(decoder1)
		outputs = Dense(vocab_size, activation='softmax')(decoder2)
		model = Model(inputs=[inputs1, inputs2], outputs=outputs)
		print('---------------------------------Model Summary--------------------------\n',self.model.summary())

		self.model = model
		#initializing embedding layer matrix
		self.model.layers[4].set_weights([self.embedding_matrix])
		self.model.layers[4].trainable = False

		print('----------------Building the model-------------------------')
		self.model.compile(loss='categorical_crossentropy', optimizer='adam')
		self.model.optimizer.lr = 0.0001

		print('--------------------Model Compiled-------------------------')


	def data_generator(seq_, dict_encoding, wordtoix, max_length,vocab_size,num_images_per_batch):
	    X1, X2, y = list(), list(), list()
	    n=0

	    while True:

	      for key,tokenized_sentence in seq_.items():

	        n+=1

	        seq = [wordtoidx[word] for word in tokenized_sentence]
	        #print(seq)
	        for i in range(1,len(seq)-1):

	          img = dict_encoding[f'{fol_name}_{video_name}_word{i-1}_{mode}']
	          in_seq, out_seq = seq[:i], seq[i]
	          #print(in_seq)
	          #print(out_seq)
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
	  
	      #print('Executing break')
	      #break


	def train(self):

		#callbacks_path = '/content/trained_weights/' + 'weights.best.{epoch:02d}-{loss:.2f}.hdf5'
		#checkpoints = ModelCheckpoint(callbacks_path, monitor='loss', verbose=1, save_weights_only=True, mode='min', period=1)
		#callbacks_list = [checkpoints]

		tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
		                                min_delta=0, 
		                                patience=0, 
		                                verbose=0, 
		                                mode='min', 
		                                baseline=None, 
		                                restore_best_weights=True)

		train_generator = data_generator(self.seq_train, self.train_encodings, self.wordtoidx, self.max_length,len(self.vocab),self.number_pics_per_batch)
		val_generator = data_generator(self.seq_val, self.val_encodings, self.wordtoidx, self.max_length,len(self.vocab),self.number_pics_per_batch)
		history = model.fit(train_generator, validation_data=val_generator, validation_steps=self.validation_steps, epochs=self.epochs, steps_per_epoch=self.steps_per_epoch)


	def lossPlot(self,history):



	def cal_WER(self):

		count = 0
		WER_loss = 0

		for t in c_test:
		  num_words = c_test[t]
		  if(num_words < 7):
		    continue
		  name,fol = t
		  word_c = 0
		  photo = []
		  for i in range(num_words):
		    img = test_encoding[f'{fol}_{name}_word{i}_test']
		    #print(img.shape,f'{fol}_{name}_word{i}_test')
		    img = np.resize(img,(15,1,2,512))
		    photo.append(img)

		  print('-'*40)
		  pred = greedySearch(photo)
		  print('Predicted -->',pred)
		  alignment = read_align(f'/content/unseen_speakers/{fol}/align/{name}.align')
		  true = ' '.join([word for _,_,word in alignment if word !='sil'])
		  print('Actual -->',true)
		  if(count == 20):
		    break
		  print('-'*40)
		  count +=1

	def predict(self,photo):

  
	    in_text = 'start'
	    j = 0
	    for i in range(self.max_length):
	        sequence = [self.wordtoidx[w] for w in in_text.split() if w in self.wordtoidx]
	        sequence = pad_sequences([sequence], maxlen=max_length)

	        if(j == len(photo)):
	          j = len(photo) - 1
	        
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


	def eval(self):

		fp = open('predictions.txt','a+') 
		for fol in os.listdir('/content/unseen_speakers/'):

			s_path = os.path.join('/content/unseen_speakers',fol)

			for file in os.listdir(s_path):

				fname,fxt = os.path.splitext(file)
				word_count = 0
				photo = []
				mode = 'test'
				key = f'{fol}_{fname}_word{word_count}_{mode}'
				while key in self.dict_encoding:
					photo.append(self.dict_encoding[key])
					word_count+=1

				y_pred = self.predict(photo)
				y_true = self.sequence[f'{fol}_{fname}_{mode}']
				fp.write(f'{fol}_{fname}|{y_pred}|{y_true}')


		fp.close()