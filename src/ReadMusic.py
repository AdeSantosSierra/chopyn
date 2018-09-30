#!/usr/bin/env python
# -*- coding: utf-8 -*-

from MusicalNote import *
from melody import *
from Play import CSVtoMIDI
from Tonality import *

import numpy as np
import pandas as pd


class Score(object):

	pass


class Read(Score):

	def __init__(self, name_file_midi):

		# Read midi file
		self.music_df = pd.read_csv(name_file_midi)

	def get_music_data(self):
		return self.music_df

	


	def get_most_common_note(self):

		# Names can be F3#, A2#, then return F# and A#
		# Names can be F3, A2, then return F and A
		self.music_df['name_note'] = \
		(self.music_df
		 .apply(lambda x:
		        _get_note_name_without_octave(x['fullNoteOctave']), axis=1)
		 )


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

def _get_note_name_without_octave(fullNoteOctave):
	# Function to get the name, regardless the octave
	if len(fullNoteOctave) == 3:
		# Names can be F3#, A2#, then return F# and A#
		return fullNoteOctave[::2]
	else:
		# Names can be F3, A2, then return F and A
		return fullNoteOctave[0]

if __name__ == "__main__":
	name_file_midi = '../../scores/Debussy_Claire_de_Lune.csv'
	chopin = Read(name_file_midi)
	# print(chopin.get_music_data().head())
	print(chopin.get_most_common_note())


