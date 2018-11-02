#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

from MusicalNote import *
from melody import *
from Polyphony import *
from Play import CSVtoMIDI
from Tonality import *
from ReadMusic import *

import numpy as np
import pandas as pd

from collections import Counter, OrderedDict
import itertools

import logging
FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel('INFO')


#LSTM Packages
import tensorflow as tf
from tensorflow.contrib import rnn
import random
import time
import os

class CreateMusicFromChords(object):



	def __init__(self, music_data, training_iters, n_input):

		self.training_iters = training_iters
		self.display_step = 1000
		self.n_input = n_input

		# Read musical data
		self.training_data = music_data['grades']

		# Target log path
		path_logs = '../tmp'
		self.writer = tf.summary.FileWriter(path_logs)

		# Extract alphabet dictionary
		alphabet = np.unique(self.training_data)
		self.dictionary = dict(zip(alphabet,range(len(alphabet))))


		

	def config_LSTM(self):
		# Parameters
		learning_rate = 0.001

		# number of units in RNN cell
		n_hidden = 1024
		vocab_size = len(self.dictionary)

		# tf Graph input
		self.x = tf.placeholder("float", [None, self.n_input, 1], name = 'x')
		self.y = tf.placeholder("float", [None, vocab_size])


		# RNN output node weights and biases
		weights = {
		    'out': tf.Variable(tf.random_normal([n_hidden, vocab_size]))
		}
		biases = {
		    'out': tf.Variable(tf.random_normal([vocab_size]))
		}

		pred = self.RNN(self.x, weights, biases, n_hidden)

		# Loss and optimizer
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=self.y), name='cost')
		optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

		# Model evaluation
		correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(self.y,1))
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

		# Initializing the variables
		self.init = tf.global_variables_initializer()
		self.saver = tf.train.Saver()

		return optimizer, accuracy, cost, pred

	def train(self, optimizer, accuracy, cost, pred, name_model):

		# Launch the graph
		with tf.Session() as session:
		    session.run(self.init)
		    self.saver.save(session, name_model)
		    step = 0
		    offset = random.randint(0,self.n_input+1)
		    end_offset = self.n_input + 1
		    acc_total = 0
		    loss_total = 0

		    vocab_size = len(self.dictionary)

		    reverse_dictionary = dict(zip(self.dictionary.values(),
		                                  self.dictionary.keys()))

		    self.writer.add_graph(session.graph)

		    while step < self.training_iters:
		        # Generate a minibatch. Add some randomness on selection process.
		        if offset > (len(self.training_data)-end_offset):
		            offset = random.randint(0, self.n_input+1)

		        symbols_in_keys = ([[self.dictionary[self.training_data[i]]] 
		                           for i in range(offset, offset+self.n_input) ])
		        symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, self.n_input, 1])

		        symbols_out_onehot = np.zeros([vocab_size], dtype=float)
		        symbols_out_onehot[self.dictionary[self.training_data[offset+self.n_input]]] = 1.0
		        symbols_out_onehot = np.reshape(symbols_out_onehot,[1,-1])

		        _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred], \
		                                                feed_dict={self.x: symbols_in_keys, self.y: symbols_out_onehot})

		        #print('pred')
		        #print(pred)
		        loss_total += loss
		        acc_total += acc
		        if (step+1) % self.display_step == 0:
		            print("Iter= " + str(step+1) + ", Average Loss= " + \
		                  "{:.6f}".format(loss_total/self.display_step) + ", Average Accuracy= " + \
		                  "{:.2f}%".format(100*acc_total/self.display_step))
		            acc_total = 0
		            loss_total = 0
		            symbols_in = [self.training_data[i] for i in range(offset, offset + self.n_input)]
		            symbols_out = self.training_data[offset + self.n_input]
		            symbols_out_pred = reverse_dictionary[int(tf.argmax(onehot_pred, 1).eval())]
		            print("%s - [%s] vs [%s]" % (symbols_in,symbols_out,symbols_out_pred))
		            self.saver.save(session, name_model, global_step=step+1)
		        step += 1
		        offset += (self.n_input+1)

	def train_and_predict(self, optimizer, accuracy, cost, pred, name_model, sequence_length, starting_sequence):

		# Launch the graph
		with tf.Session() as session:
		    session.run(self.init)
		    self.saver.save(session, name_model)
		    step = 0
		    offset = random.randint(0,self.n_input+1)
		    end_offset = self.n_input + 1
		    acc_total = 0
		    loss_total = 0

		    vocab_size = len(self.dictionary)

		    reverse_dictionary = dict(zip(self.dictionary.values(),
		                                  self.dictionary.keys()))

		    self.writer.add_graph(session.graph)

		    while step < self.training_iters:
		        # Generate a minibatch. Add some randomness on selection process.
		        if offset > (len(self.training_data)-end_offset):
		            offset = random.randint(0, self.n_input+1)

		        symbols_in_keys = ([[self.dictionary[self.training_data[i]]] 
		                           for i in range(offset, offset+self.n_input) ])
		        symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, self.n_input, 1])

		        symbols_out_onehot = np.zeros([vocab_size], dtype=float)
		        symbols_out_onehot[self.dictionary[self.training_data[offset+self.n_input]]] = 1.0
		        symbols_out_onehot = np.reshape(symbols_out_onehot,[1,-1])

		        _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred], \
		                                                feed_dict={self.x: symbols_in_keys, self.y: symbols_out_onehot})

		        #print('pred')
		        #print(pred)
		        loss_total += loss
		        acc_total += acc
		        if (step+1) % self.display_step == 0:
		            print("Iter= " + str(step+1) + ", Average Loss= " + \
		                  "{:.6f}".format(loss_total/self.display_step) + ", Average Accuracy= " + \
		                  "{:.2f}%".format(100*acc_total/self.display_step))
		            acc_total = 0
		            loss_total = 0
		            symbols_in = [self.training_data[i] for i in range(offset, offset + self.n_input)]
		            symbols_out = self.training_data[offset + self.n_input]
		            symbols_out_pred = reverse_dictionary[int(tf.argmax(onehot_pred, 1).eval())]
		            print("%s - [%s] vs [%s]" % (symbols_in,symbols_out,symbols_out_pred))
		            self.saver.save(session, name_model, global_step=step+1)
		        step += 1
		        offset += (self.n_input+1)

			symbols_in_keys = [self.dictionary[(iter_sequence)] for iter_sequence in starting_sequence]
		    output_sequence = list()

		    for i in range(sequence_length):
				keys = np.reshape(symbols_in_keys, [-1, self.n_input, 1])
				onehot_pred = session.run(pred, feed_dict={self.x: keys})
				onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
				print('symbols_out')
				print(onehot_pred_index)
				output_sequence.append(reverse_dictionary[onehot_pred_index])
				symbols_in_keys = symbols_in_keys[1:]
				symbols_in_keys.append(onehot_pred_index)
				print('symbols_in_keys')
				print(symbols_in_keys)

		return output_sequence

	def load_and_predict (self, dir_name_model, model_metadata, starting_sequence, sequence_length):

		output_sequence = list()

		with tf.Session() as session:

			# Other attempts
			# tf.saved_model.loader.load(session,[name_model], '/tmp')

			#First let's load meta graph and restore weights
			saver = tf.train.import_meta_graph(model_metadata)

			# Initialize variables
			session.run(tf.global_variables_initializer())
			saver.restore(session,tf.train.latest_checkpoint(dir_name_model))
			
			graph = tf.get_default_graph()
			pred  = graph.get_tensor_by_name("pred:0")
			x     = graph.get_tensor_by_name("x:0")




			reverse_dictionary = dict(zip(self.dictionary.values(),
			                                  self.dictionary.keys()))

			symbols_in_keys = [self.dictionary[(iter_sequence)] for iter_sequence in starting_sequence]
			

			for i in range(sequence_length):
				keys              = np.reshape(np.array(symbols_in_keys), [-1, self.n_input, 1])
				onehot_pred       = session.run(pred, feed_dict={x: keys})
				onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
				# print('symbols_out')
				# print(onehot_pred_index)
				output_sequence.append(reverse_dictionary[onehot_pred_index])
				symbols_in_keys = symbols_in_keys[1:]
				symbols_in_keys.append(onehot_pred_index)
				# print('symbols_in_keys')
				# print(symbols_in_keys)

		return output_sequence



	def RNN(self, x, weights, biases, n_hidden):
	    
	    # reshape to [1, n_input]
	    x = tf.reshape(x, [-1, self.n_input])

	    # Generate a n_input-element sequence of inputs
	    # (eg. [had] [a] [general] -> [20] [6] [33])
	    x = tf.split(x,self.n_input,1)

	    # 2-layer LSTM, each layer has n_hidden units.
	    # Average Accuracy= 95.20% at 50k iter
	    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden),
	                                rnn.BasicLSTMCell(n_hidden),
	                                rnn.BasicLSTMCell(n_hidden)])

	    # generate prediction
	    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

	    # there are n_input outputs but
	    # we only want the last output
	    pred = tf.matmul(outputs[-1], weights['out']) + biases['out']
	    tf.identity(pred, 'pred')
	    return pred


class PlayMusicFromChords(object):

	def __init__(self, name_file_midi, 
	             n_input = 20, 
	             training_iters = 100000, 
	             sequence_length = 500, 
	             model_version_to_load = 99000, 
	             bool_train = False):

		musical_piece = Read(name_file_midi)

		print('La tonalidad es: '+musical_piece.get_tonality())

		logger.info('Calculate the tonality and apply it to the whole music piece')
		grades_chords = musical_piece.apply_tonality()

		logger.info('Extract the sequence of chords')


		name_model = 'n_input_'+str(n_input)+'_chromatic'+'_iters_'+str(training_iters)+'_'+name_file_midi[13:-4]
		dir_name_model = '../models/'+name_model

		if not os.path.exists(dir_name_model):
		    os.makedirs(dir_name_model)


		logger.info('Create the Deep Learning object')
		music_creator = CreateMusicFromChords(grades_chords,
		                                      training_iters = training_iters,
		                                      n_input = n_input
		                                      )	

		
		if bool_train:
			logger.info('Config LSTM')
			optimizer, accuracy, cost, pred = music_creator.config_LSTM()


		logger.info('Estimate initial sequence to predict based on LSTM')
		grades_chords_values = grades_chords['grades']
		initial_point = random.randint(0,len(grades_chords_values)-n_input-1)
		initial_sequence_chords = list(grades_chords_values
		                               [initial_point:(initial_point+n_input)
		                               ]
		                               )

		if bool_train:
			logger.info('Train and save LSTM')
			music_creation = \
			music_creator.train(optimizer, accuracy, cost, pred, dir_name_model+'/'+name_model,
			                                #sequence_length = sequence_length,
			                                #starting_sequence = initial_sequence_chords
			                                )

		logger.info('Create Music!!')
		music_creation = \
		music_creator.load_and_predict(dir_name_model,
		                               dir_name_model+'/'+name_model+'-'+str(model_version_to_load)+'.meta',
		                               initial_sequence_chords,
		                               sequence_length = sequence_length
		                               )

		logger.info('Convert grades to sequences')
		chords_notes = (musical_piece
		                .convert_grades_sequence_to_notes(music_creation,
		                                                  musical_piece.get_tonality()
		                                                  )
		                )


		logger.info('Convert it to MIDI')
		polyphony = SequenceChordPolyphony(chords_notes)
		CSVtoMIDI(polyphony
		          .convert_to_midi(),
		          'polyphony_'+name_file_midi[13:-4]
		          )

		logger.info('Finished!!!')


class CreateMusicFromDataframe(object):

	def __init__(self, music_data, training_iters, n_input):

		self.training_iters = training_iters
		self.display_step = 1000
		self.n_input = n_input

		# Read musical data
		self.training_data = music_data
		self.num_columns_training_data = self.training_data.shape[1]

		# Target log path
		path_logs = '../tmp'
		self.writer = tf.summary.FileWriter(path_logs)


	def config_LSTM(self):
		# Parameters
		learning_rate = 0.001

		# number of units in RNN cell
		n_hidden = 1024

		# tf Graph input
		self.x = tf.placeholder(dtype=tf.float32, 
		                        shape=(None, self.num_columns_training_data), 
		                        name = 'x')
		self.y = tf.placeholder(dtype=tf.float32, 
		                        shape=(None, self.num_columns_training_data),
		                        name = 'y'
		                        )

		# RNN output node weights and biases
		weights = {
		    'out': tf.Variable(tf.random_uniform([n_hidden, self.num_columns_training_data], 
		                                         minval = 0, 
		                                         maxval = 1000, dtype=tf.float32))
		}
		biases = {
		    'out': tf.Variable(tf.random_uniform([self.num_columns_training_data], 
		                                         minval = 0, 
		                                         maxval = 1000, dtype=tf.float32))
		}

		pred = self.RNN(self.x, weights, biases, n_hidden)

		# Loss and optimizer
		# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, 
		#                                                               labels=self.y), name='cost')
		# pred = tf.Print(pred, [pred], 
		#                 message="This is pred: ", 
		#                 summarize = 100)
		# pred = tf.Print(pred, [tf.shape(pred)], message="This is pred: ")

		cost = tf.reduce_sum(tf.square(self.y - pred))


		optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

		# Model evaluation
		# correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(self.y,1))
		# correct_pred = tf.cast(pred, tf.int32)
		# correct_pred = tf.Print(correct_pred, [correct_pred], message="This is correct_pred: ")
		accuracy     = cost
		
		# Initializing the variables
		self.init    = tf.global_variables_initializer()
		self.saver   = tf.train.Saver()

		return optimizer, accuracy, cost, pred

	def RNN(self, x, weights, biases, n_hidden):

		# reshape to [1, n_input]
		# -1 means to be inferred
		# tensor 't' is [[[1, 1, 1],
		#                 [2, 2, 2]],
		#                [[3, 3, 3],
		#                 [4, 4, 4]],
		#                [[5, 5, 5],
		#                 [6, 6, 6]]]
		# tensor 't' has shape [3, 2, 3]

		# -1 is inferred to be 2:
		# reshape(t, [-1, 9]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
		#                          [4, 4, 4, 5, 5, 5, 6, 6, 6]]
		
		x = tf.reshape(x, [-1, self.num_columns_training_data])

	    # Generate a n_input-element sequence of inputs
	    # (eg. [had] [a] [general] -> [20] [6] [33])
	    # 0 means per n_input (horizontal dimension)
		x = tf.split(x, self.n_input, 0)

	    # # reshape to [1, n_input]
	    # x = tf.reshape(x, [-1, self.n_input])

	    # # Generate a n_input-element sequence of inputs
	    # # (eg. [had] [a] [general] -> [20] [6] [33])
	    # x = tf.split(x,self.n_input,1)

	    # 2-layer LSTM, each layer has n_hidden units.
	    # Average Accuracy= 95.20% at 50k iter
		rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden),
	                                rnn.BasicLSTMCell(n_hidden),
	                                rnn.BasicLSTMCell(n_hidden)])

	    # generate prediction
		outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

	    # there are n_input outputs but
	    # we only want the last output
		pred = tf.matmul(outputs[-1], weights['out']) + biases['out']
		tf.identity(pred, 'pred')
		return pred

	def train(self, optimizer, accuracy, cost, pred, name_model):

		# Launch the graph
		with tf.Session() as session:
		    session.run(self.init)
		    self.saver.save(session, name_model)
		    step = 0
		    offset = random.randint(0,self.n_input+1)
		    end_offset = self.n_input + 1
		    acc_total = 0
		    loss_total = 0

		    # vocab_size = len(self.dictionary)

		    # reverse_dictionary = dict(zip(self.dictionary.values(),
		    #                               self.dictionary.keys()))

		    self.writer.add_graph(session.graph)

		    while step < self.training_iters:
		        # Generate a minibatch. Add some randomness on selection process.
		        if offset > (len(self.training_data)-end_offset):
		            offset = random.randint(0, self.n_input+1)

		        # symbols_in_keys = ([[self.dictionary[self.training_data[i]]] 
		        #                    for i in range(offset, offset+self.n_input) ])
		        # symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, self.n_input, 1])

		        # symbols_out_onehot = np.zeros([vocab_size], dtype=float32float)
		        # symbols_out_onehot[self.dictionary[self.training_data[offset+self.n_input]]] = 1.0
		        # symbols_out_onehot = np.reshape(symbols_out_onehot,[1,-1])

		        input_x = self.training_data.loc[offset:(offset+self.n_input-1),:]
		        input_y = self.training_data.loc[(offset+self.n_input-1),:].to_frame().T

		        _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred], \
		                                                feed_dict={self.x: input_x, self.y: input_y})

		        #print('pred')
		        #print(pred)
		        loss_total += loss
		        acc_total += acc
		        if (step+1) % self.display_step == 0:
		            print("Iter= " + str(step+1) + ", Average Loss= " + \
		                  "{:.6f}".format(loss_total/self.display_step) + ", Average Accuracy= " + \
		                  "{:.2f}%".format(100*acc_total/self.display_step))
		            acc_total = 0
		            loss_total = 0
		            # symbols_in = [self.training_data[i] for i in range(offset, offset + self.n_input)]
		            # symbols_out = self.training_data[offset + self.n_input]
		            # symbols_out_pred = reverse_dictionary[int(tf.argmax(onehot_pred, 1).eval())]
		            # print("%s - [%s] vs [%s]" % (symbols_in,symbols_out,symbols_out_pred))
		            self.saver.save(session, name_model, global_step=step+1)
		        step += 1
		        offset += (self.n_input+1)



if __name__ == '__main__':

	# python ../../Data\ Beers/src/midicsv-process.py Gymnopedie_No_1.midi > Gymnopedie_No_1.csv
	name_file_midi = '../../scores/Schubert_S560_Schwanengesang_no7.csv'
	name_file_midi = '../../scores/Brahms_symphony_2_2.csv' # Si M
	name_file_midi = '../../scores/Albeniz_Asturias.csv' # Doesn't detect properly 
	name_file_midi = '../../scores/Chopin_Etude_Op_10_n_5.csv'
	name_file_midi = '../../scores/Schuber_Impromptu_D_899_No_3.csv'
	name_file_midi = '../../scores/Mozart_Rondo.csv'
	name_file_midi = '../../scores/Chopin_Etude_Op_10_n_1.csv'
	name_file_midi = '../../scores/Mozart_Sonata_16.csv'
	name_file_midi = '../../scores/Bach-Partita_No1_in_Bb_BWV825_7Gigue.csv'
	name_file_midi = '../../scores/Brahms_symphony_2_1.csv'
	name_file_midi = '../../scores/Bach_Cello_Suite_No_1.csv'
	name_file_midi = '../../scores/Gymnopedie_No_1.csv'
	name_file_midi = '../../scores/Debussy_Claire_de_Lune.csv'
	#name_file_midi = '../../scores/Beethoven_Moonlight_Sonata_third_movement.csv'
	#name_file_midi = '../../scores/Schubert_Piano_Trio_2nd_Movement.csv'
	
	# PlayMusicFromChords(name_file_midi, 
	#                     n_input = 20, 
	#                     training_iters = 100000, 
	#                     sequence_length = 500, 
	#                     model_version_to_load = 99000, 
	#                     bool_train = False)

	musical_piece = Read(name_file_midi)

	print('La tonalidad es: '+musical_piece.get_tonality())

	logger.info('Obtain the main dataframe of the musical piece')
	musical_dataframe = musical_piece.convert_tonality_to_music_dataframe()


	music_creator = CreateMusicFromDataframe(musical_dataframe,
		                                     training_iters = 100000,
		                                     n_input = 10
		                                     )	

	logger.info('Config LSTM')
	optimizer, accuracy, cost, pred = music_creator.config_LSTM()
	music_creation = \
	music_creator.train(optimizer, accuracy, cost, pred, name_model = '../models/'+'prueba.modelo',
	                                #sequence_length = sequence_length,
	                                #starting_sequence = initial_sequence_chords
	                                )

