#!/usr/bin/env python
# -*- coding: utf-8 -*-

from MusicalNote import *
from Play import CSVtoMIDI

import numpy as np 
import pandas as pd

# Random Melody
# A Melody is a sequence of notes


class Melody (object):



	def convert_to_midi(self):

		start_ms_array = \
		(np.cumsum([0]+[note_from_melody.get_duration() 
			for note_from_melody in self.sequence_of_notes]))

		# Create dataframe with musical properties from sequence/melody
		# print([note_from_melody.get_play_props() for note_from_melody in self.sequence_of_notes])
		# print('pppppppp')
		melody_dataframe = pd.DataFrame.from_records([note_from_melody.get_play_props() 
			for note_from_melody in self.sequence_of_notes])

		# print(melody_dataframe)

		# Add the time sequence
		melody_dataframe['start_ms'] = start_ms_array[:-1]

		# Rename columns as the MIDI has already specific names for the columns
		# print(melody_dataframe.columns)
		melody_dataframe.columns = ['dur_ms','velocity','pitch','part','start_ms']

		return melody_dataframe	

class RandomMelody(Melody):

	def __init__(self):

		notes_props = ({'duration':np.random.randint(500,1000), 
						'intensity':np.random.randint(50),
						'timbre':np.random.randint(10)})

		notes_names = [Do,Re,Mi,Fa,Sol,La,Si]
		melody_length = 20
		random_sequence = np.random.choice(notes_names, melody_length, replace=True)

		self.sequence_of_notes = [iterator_note(**{'duration':np.random.randint(500,1000), 
												   'intensity':70,
												   'timbre':1}) 
							      for iterator_note in random_sequence]


class SequenceMelody(Melody):
	def __init__(self,sequence_of_notes):
		self.sequence_of_notes = sequence_of_notes


if __name__ == '__main__':
	melody = RandomMelody()
	CSVtoMIDI(melody.convert_to_midi())

