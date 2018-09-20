#!/usr/bin/env python
# -*- coding: utf-8 -*-

from MusicalNote import *
from Play import CSVtoMIDI

import numpy as np 
import pandas as pd

# Random Melody
# A Melody is a sequence of notes

class RandomMelody(object):

	def __init__(self):

		notes_props = ({'duration':10*np.random.randint(100), 
						'intensity':np.random.randint(50),
						'timbre':np.random.randint(10)})

		notes_names = [Do,Re,Mi,Fa,Sol,La,Si]
		melody_length = 30
		random_sequence = np.random.choice(notes_names, melody_length, replace=True)

		self.sequence_of_notes = [iterator_note(**notes_props) for iterator_note in random_sequence]

		

	def convert_to_midi(self):

		start_ms_array = \
		(np.cumsum([0]+[note_from_melody.get_duration() 
			for note_from_melody in self.sequence_of_notes]))

		# Create dataframe with musical properties from sequence/melody
		melody_dataframe = pd.DataFrame.from_records([note_from_melody.get_props() 
			for note_from_melody in self.sequence_of_notes])

		# Add the time sequence
		melody_dataframe['start_ms'] = start_ms_array[:-1]

		# Rename columns as the MIDI has already specific names for the columns
		melody_dataframe.columns = ['dur_ms','velocity','pitch','part','start_ms']

		return melody_dataframe	





if __name__ == '__main__':
	melody = RandomMelody()
	CSVtoMIDI(melody.convert_to_midi())

