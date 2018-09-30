#!/usr/bin/env python
# -*- coding: utf-8 -*-

from MusicalNote import *
from melody import *
from Play import CSVtoMIDI
from Tonality import *

import numpy as np
import pandas as pd

from collections import Counter


class Score(object):

	pass


class Read(Score):

	def __init__(self, name_file_midi):

		# Read midi file
		self.music_df = pd.read_csv(name_file_midi)
		# Calculate attribute name_note
		self.calculate_name_note()

	def get_music_data(self):
		return self.music_df

	def calculate_name_note(self):

		# Names can be F3#, A2#, then return F# and A#
		# Names can be F3, A2, then return F and A
		self.music_df['name_note'] = \
		(self.music_df
		 .apply(lambda x:
		        _get_note_name_without_octave(x['fullNoteOctave']), axis=1)
		 )


	def get_most_common_note(self):

		return (self
		      .music_df
		      # Group by name of the note (A, F#, ...)
		      .groupby(['name_note'])
		      # Take the total length that note has been played
		      .agg({'dur_ms':sum})
		      .rename(columns={'dur_ms': 'cum_duration'})
		      # Order in descending manner
		      .sort_values(['cum_duration'],ascending=False)
		      .reset_index()
		      # Take the first element and the corresponding column
		      .head(1)['name_note'][0]
		      )

	def get_most_granular_tick(self):

		print((Counter(self.music_df['start_ticks'].diff())))
		print(np.sort(Counter(self.music_df['start_ticks'].diff()).keys()))

	def get_chord_from_tick(self):
		# Given a sequence of notes such as:
		# 0,0,79,85,90,F6#,96,1
		# 0,0,119,128,66,F4#,96,2
		# 0,0,119,128,61,C4#,96,2
		# 0,0,119,128,58,A3#,96,2
		# 0,0,119,128,54,F3#,96,2
		# 80,86,79,85,94,A6#,96,1
		# 160,172,79,85,85,C6#,96,1
		# 240,258,79,85,90,F6#,96,1
		# 240,258,119,128,66,F4#,96,2
		# 240,258,119,128,63,D4#,96,2
		# 240,258,119,128,59,B3,96,2
		# 240,258,119,128,54,F3#,96,2

		# This method will extract the chords for 0, 80, 160, 240 and so forth.
		print(self.music_df
		      .groupby('start_ticks')
		      .agg({'fullNoteOctave':lambda x: Counter(x),
		           'name_note':lambda x: Counter(x)}
		           )
		      .reset_index()
		      )

		

def _get_note_name_without_octave(fullNoteOctave):
	# Function to get the name, regardless the octave
	if len(fullNoteOctave) == 3:
		# Names can be F3#, A2#, then return F# and A#
		return fullNoteOctave[::2]
	else:
		# Names can be F3, A2, then return F and A
		return fullNoteOctave[0]

if __name__ == "__main__":
	name_file_midi = '../../scores/Chopin_Etude_Op_10_n_5.csv'
	chopin = Read(name_file_midi)
	# print(chopin.get_music_data().head())
	print(chopin.get_most_common_note())
	print(chopin.get_chord_from_tick())


