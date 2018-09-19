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
		
		notes_props = {'duration':500,'intensity':100,'timbre':1}
		long_notes_props = {'duration':1000,'intensity':100,'timbre':1}
		short_notes_props = {'duration':250,'intensity':100,'timbre':1}

		self.sequence_of_notes = [Do(**notes_props), Do(**notes_props),
		Sol(**notes_props), Sol(**notes_props),
		La(**notes_props), La(**notes_props),
		Sol(**long_notes_props),
		Fa(**notes_props), Fa(**notes_props),
		Mi(**notes_props), Mi(**notes_props),
		Re(**notes_props), Re(**notes_props),
		Do(**long_notes_props)
		]

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

