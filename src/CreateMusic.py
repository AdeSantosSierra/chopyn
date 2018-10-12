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

class CreateMusicFromChords(object):

	def __init__(self, path_chords_sequence_file):
		# Read musical data
		music_data = pd.read_csv(path_chords_sequence_file)
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
		self.training_iters = 10000
		self.display_step = 1000
		self.n_input = 10

		# number of units in RNN cell
		n_hidden = 512
		vocab_size = len(self.dictionary)

		# tf Graph input
		self.x = tf.placeholder("float", [None, self.n_input, 1])
		self.y = tf.placeholder("float", [None, vocab_size])


		# RNN output node weights and biases
		weights = {
		    'out': tf.Variable(tf.random_normal([n_hidden, vocab_size]))
		}
		biases = {
		    'out': tf.Variable(tf.random_normal([vocab_size]))
		}

		self.saver = tf.train.Saver()


		pred = self.RNN(self.x, weights, biases, n_hidden)

		# Loss and optimizer
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=self.y))
		optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

		# Model evaluation
		correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(self.y,1))
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

		# Initializing the variables
		self.init = tf.global_variables_initializer()

		return optimizer, accuracy, cost, pred

	def train(self, optimizer, accuracy, cost, pred):

		name_model = '../models/my_first_model'
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

		        symbols_in_keys = ([[self.dictionary[ str(self.training_data[i])]] 
		                           for i in range(offset, offset+self.n_input) ])
		        symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, self.n_input, 1])

		        symbols_out_onehot = np.zeros([vocab_size], dtype=float)
		        symbols_out_onehot[self.dictionary[str(self.training_data[offset+self.n_input])]] = 1.0
		        symbols_out_onehot = np.reshape(symbols_out_onehot,[1,-1])

		        _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred], \
		                                                feed_dict={self.x: symbols_in_keys, self.y: symbols_out_onehot})
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
		        step += 1
		        offset += (self.n_input+1)
		    	self.saver.save(session, name_model,global_step=1000)

	def load (self, model_metadata):
		session=tf.Session()    
		#First let's load meta graph and restore weights
		saver = tf.train.import_meta_graph(model_metadata)
		saver.restore(session,tf.train.latest_checkpoint('../models/'))



	def RNN(self, x, weights, biases, n_hidden):
	    
	    # reshape to [1, n_input]
	    x = tf.reshape(x, [-1, self.n_input])
	    print(x)

	    # Generate a n_input-element sequence of inputs
	    # (eg. [had] [a] [general] -> [20] [6] [33])
	    x = tf.split(x,self.n_input,1)
	    print(x)

	    # 2-layer LSTM, each layer has n_hidden units.
	    # Average Accuracy= 95.20% at 50k iter
	    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden),rnn.BasicLSTMCell(n_hidden)])
	    print(rnn_cell)
	    
	    # 1-layer LSTM with n_hidden units but with lower accuracy.
	    # Average Accuracy= 90.60% 50k iter
	    # Uncomment line below to test but comment out the 2-layer rnn.MultiRNNCell above
	    # rnn_cell = rnn.BasicLSTMCell(n_hidden)

	    # generate prediction
	    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)
	    print(outputs)

	    # there are n_input outputs but
	    # we only want the last output
	    return tf.matmul(outputs[-1], weights['out']) + biases['out']


if __name__ == '__main__':


	name_file_midi = '../../scores/Schubert_S560_Schwanengesang_no7.csv'
	name_file_midi = '../../scores/Brahms_symphony_2_2.csv' # Si M
	name_file_midi = '../../scores/Brahms_symphony_2_1.csv'
	name_file_midi = '../../scores/Bach-Partita_No1_in_Bb_BWV825_7Gigue.csv'
	name_file_midi = '../../scores/Albeniz_Asturias.csv'
	name_file_midi = '../../scores/Chopin_Etude_Op_10_n_5.csv'
	name_file_midi = '../../scores/Schuber_Impromptu_D_899_No_3.csv'
	name_file_midi = '../../scores/Debussy_Claire_de_Lune.csv'
	name_file_midi = '../../scores/Chopin_Etude_Op_10_n_1.csv'
	#name_file_midi = '../../scores/Beethoven_Moonlight_Sonata_third_movement.csv'
	#name_file_midi = '../../scores/Schubert_Piano_Trio_2nd_Movement.csv'
	
	chopin = Read(name_file_midi)
	# print(chopin.get_music_data().head())
	#print(chopin.get_chord_from_tick().filter(['fullNoteOctave']))
	print('La tonalidad es: '+chopin.get_tonality())
	name_grades_chords = '../tmp/'+name_file_midi[13:-4]+'_grades_chords.csv'
	# grades_chords = chopin.apply_tonality()
	# grades_chords.to_csv(name_grades_chords,
	#                      header=True,
	#                      index_label=None)

	music_creator = CreateMusicFromChords(name_grades_chords)	
	optimizer, accuracy, cost, pred = music_creator.config_LSTM()
	music_creator.train(optimizer, accuracy, cost, pred)
	music_creator.load('../models/my_first_model.meta')

