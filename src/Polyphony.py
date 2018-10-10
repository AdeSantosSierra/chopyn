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

		individual_notes = list()
		chord_id = list()

		duration_length = 250
		factor_duration = 2

		for id_iter_chord, iter_chord in enumerate(self.sequenceChord):
			for part_note_in_chord, iter_note_in_chord in enumerate(iter_chord):
				iter_note_in_chord.update_props({'timbre':part_note_in_chord+1})
				individual_notes.append(iter_note_in_chord)
				# Store the id of the chord to know which notes are together
				chord_id.append((id_iter_chord+1)*factor_duration*duration_length)


		# Create dataframe with musical properties from sequence/melody
		# print([note_from_melody.get_play_props() for note_from_melody in self.sequence_of_notes])
		# print('pppppppp')
		music_dataframe = pd.DataFrame.from_records([note_from_melody.get_play_props() 
			for note_from_melody in individual_notes])

		music_dataframe['start_ms'] = chord_id
		music_dataframe['duration'] = duration_length*factor_duration

		# Rename columns as the MIDI has already specific names for the columns
		music_dataframe.columns = ['dur_ms','velocity','pitch','part','start_ms']

		return music_dataframe	



class SequenceChordPolyphony (Polyphony):

	def __init__(self, sequenceChord):
		self.sequenceChord = sequenceChord