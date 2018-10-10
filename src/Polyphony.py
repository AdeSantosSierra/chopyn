#!/usr/bin/env python
# -*- coding: utf-8 -*-

from MusicalNote import *
from Play import CSVtoMIDI

import numpy as np 
import pandas as pd

# Random Melody
# A Melody is a sequence of notes


class Polyphony (object):

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



class SequenceChordPolyphony (Polyphony):
	pass