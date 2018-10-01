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
		self.divide_music_with_most_granular_tick()

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

	def divide_music_with_most_granular_tick(self):

		# Obtain the histograms of ticks
		# Counter({80.0: 946, 0.0: 642, 40.0: 15, 240.0: 8, 60.0: 6, 20.0: 3, 480.0: 2, 120.0: 2, nan: 1, 160.0: 1, 300.0: 1, 360.0: 1})
		hist_ticks = Counter(self.music_df['start_ticks'].diff())
		# Take the minimum_tick different from 0
		# [  0.  20.  40.  60.  80. 120. 160. 240. 300. 360. 480.  nan]
		# In this case, it is 20.
		minimum_tick = int([ticks for ticks in np.sort(hist_ticks.keys()) if ticks > 0][0])

		# Divide music with minimum_tick

		self.music_df['num_minimum_ticks'] = \
		((self.music_df['dur_ticks']+1) / minimum_tick)

		granular_music_list = list()

		for index, iter_note in self.music_df.iterrows():
			for iter_num_minimum_ticks in range(int(iter_note['num_minimum_ticks'])):
				granular_music_list.append(([iter_note['start_ticks']+(iter_num_minimum_ticks)*minimum_tick,
				                          iter_note['pitch'],iter_note['velocity'],
				                          iter_note['part'],iter_note['fullNoteOctave'],
				                          iter_note['name_note']]
				                          ))

		self.granular_music_df = (pd.DataFrame(granular_music_list,
		                                  columns = ['start_ticks','pitch','velocity','part',
		                                  'fullNoteOctave','name_note']))



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

		return (self.granular_music_df
		      .groupby('start_ticks')
		      .agg({'fullNoteOctave':lambda x: Counter(x),
		           'name_note':lambda x: Counter(x)}
		           )
		      .reset_index()
		      )
		# Do not forget the aggregated time pero granular chord.

		

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
	#name_file_midi = '../../scores/Chopin_Etude_Op_10_n_1.csv'
	chopin = Read(name_file_midi)
	# print(chopin.get_music_data().head())
	print(chopin.get_most_common_note())
	print(chopin.get_chord_from_tick())


