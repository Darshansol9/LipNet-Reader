from keras.layers.convolutional import Conv3D, ZeroPadding3D
from keras.layers.pooling import MaxPooling3D
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.recurrent import GRU
from keras.layers import Input
from keras.models import Model
from keras import backend as K
from word_embed import generateEmbeddings
from model2_datapipeline import read_data,compute_vocab,to_max_length

class CNN_LSTM:


	def __init__(self):

		self.embedding_dim = 100
		self.embedding_matrix = generateEmbeddings()
		self.dict_encoding,self.sequence,vocab_lst = read_data()
		self.vocab = compute_vocab(vocab_lst)
		self.vocab_size = len(self.vocab) + 1
		self.wordtoix,self.idxtoword = word_to_index(self.vocab)
		self.max_length = to_max_length(list(self.sequence.values()))
		self.input_size = self.dict_encoding[list(self.dict_encoding.keys())[0]].shape
		self.dense_size = 2048
		self.lstm_cell = 256
		self.dropout = 0.2
		self.epochs = 30
		self.num_videos_per_batch = 256
		self.steps = len(self.sequence)//self.num_videos_per_batch
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
		#print(self.model.summary())

		self.model = model
		#initializing embedding layer matrix
		self.model.layers[4].set_weights([self.embedding_matrix])
		self.model.layers[4].trainable = False

		#print('Building the model ....')
		self.model.compile(loss='categorical_crossentropy', optimizer='adam')
		self.model.optimizer.lr = 0.0001

		#print('Model Compiled')


	def data_generator(self,sequence, images, wordtoix, max_length,vocab_size,num_images_per_batch):
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
	          out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]

	          #taking curr image and previous word to predict next word
	          X1.append(img)
	          X2.append(in_seq)
	          y.append(out_seq)

	        #taking 256 batches at a time
	        if n==num_images_per_batch:
	          #print('Batch fullfilled')
	          yield ((np.array(X1), np.array(X2)), np.array(y))
	          X1, X2, y = list(), list(), list()
	          n=0


	def train(self):

		for i in range(epochs):
		    print(f'Epoch {i}')
		    generator = self.data_generator(self.sequence, self.dict_encoding, self.wordtoidx, self.max_length,self.vocab_size,self.number_videos_per_batch)
		    history = self.model.fit(generator, epochs=1, steps_per_epoch=self.steps, verbose=1)
		    #self.model.save('/content/model_weights/model_' + str(i) + '.h5')


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

				

	

