from PIL import Image
import pytesseract
import sympy
import pylatexenc
from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda, Embedding
from keras.optimizers import Adam, RMSprop
from keras.utils import to_categorical, plot_model
from keras.models import load_model, Model
import keras.backend as K
import numpy as np
from faker import Faker
import random
import csv
import sys  
from nltk import sent_tokenize, word_tokenize, pos_tag
from gensim.models import Word2Vec, KeyedVectors
import os
import h5py
import pydot 
import graphviz
import random
from nltk.corpus import wordnet as wn
import string

	
vocab_list = np.load("/Users/parthpatel/Desktop/vocabs2.npy")
vocab = sorted(set(vocab_list))
vocab_size = len(vocab)
#word_to_id = dict.fromkeys(vocab,1)
#id_to_word = dict.fromkeys(np.arange(vocab_size),"")
word_to_id = {}
id_to_word = {}
x = 1;
for i in vocab:
	word_to_id[i] = x
	id_to_word[x] = i
	x=x+1
id_to_word[vocab_size] = '<unk>'
word_to_id['<unk>'] = vocab_size

x = np.load("/Users/parthpatel/Desktop/chatx.npy")
y = np.load("/Users/parthpatel/Desktop/chaty.npy")
y_target = np.load("/Users/parthpatel/Desktop/chatytar.npy")



encoder_inputs = Input(shape = (20,))
embed = Embedding(input_dim=vocab_size+1,output_dim=30,mask_zero=True,input_length=20)(encoder_inputs)
encoder = Bidirectional(LSTM(256,return_state = True, recurrent_dropout=0.8, dropout=0.2))
encoder_outputs, state_h_for, state_c_for,state_h_back,state_c_back = encoder(embed) 
state_h = Concatenate()([state_h_for,state_h_back])
state_c = Concatenate()([state_c_for,state_c_back])
encoder_states = [state_h,state_c]
decoder_inputs = Input(shape = (10,))
embed2 = Embedding(input_dim=vocab_size+1,output_dim=30,mask_zero=True,input_length=10)(decoder_inputs)
decoder = LSTM(512,return_state=True, return_sequences=True, recurrent_dropout=0.8, dropout=0.2)
decoder_outputs,_,_=decoder(embed2,initial_state = encoder_states)
decoder_dense = Dense(vocab_size+1, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
model = Model(inputs=[encoder_inputs,decoder_inputs],outputs=[decoder_outputs])
opt = Adam(lr=0.01)
model.load_weights("LSTM_newWeights_au67.h5")
model.compile(optimizer=opt, loss='categorical_crossentropy')
#model.fit([x,y],y_target,epochs=4,batch_size=100, shuffle=True, validation_split = 0.05)
#model.save_weights("LSTM_newWeights_au67.h5")


encoder_model = Model(encoder_inputs, encoder_states)
decoder_state_input_h = Input(shape=(512,))
decoder_state_input_c = Input(shape=(512,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder(embed2, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs,[decoder_outputs] + decoder_states)


def decode(input_seq):
	states_vals = encoder_model.predict(input_seq)
	target_seq = np.zeros((1,10))
	target_seq[0][0] = word_to_id['S']
	sentence = ""
	end_condition = False
	while not end_condition:
		outputs,h,c = decoder_model.predict([target_seq]+states_vals)
		max_index = np.argmax(outputs[0,-1,:])
		word = id_to_word[max_index]
		if(word == "E" or len(sentence) > 100):
			end_condition = True
		sentence+=" "+word
		target_seq = np.zeros((1,10))
		target_seq[0][0] = max_index
		states_vals =[h,c]
	return sentence
def unencode(x):
	sentence = ""
	for e in x:
		if e == 0:
			break
		sentence+=" "+id_to_word[e]
	return sentence
texto = "Are you smart or nah"
sequence = np.zeros((1,20))
z=0
for e in word_tokenize(texto.lower()):
	if(z < 20):
		if(e in word_to_id):
			index = word_to_id[e]
		else:
			index = word_to_id['<unk>']
		sequence[0][z] = index
		z=z+1

print(unencode(x[1]))
print(unencode(y[1]))

exam = np.zeros((1,20))
exam[0] = x[1]
print(decode(sequence))









