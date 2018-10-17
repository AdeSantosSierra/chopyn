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

		duration_length = 125
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

		# Merge together those repeated notes
		extended_music_dataframe_list = []

		print('-------Fallan aqui los .loc')
		for a, df_gb in music_dataframe.groupby(['velocity','pitch','part']):
			df_gb.loc[:,'next_start_ms'] = ((df_gb['start_ms']+df_gb['dur_ms'])
			                          .shift(1)
			                          .fillna(0)
			                          )
			df_gb.loc[:,'diff_start_ms'] = ((df_gb['start_ms']-df_gb['next_start_ms'])>0).astype(int)
			df_gb.loc[:,'cum_sum'] = np.cumsum(df_gb['diff_start_ms'])
			# print(df_gb[['dur_ms','start_ms','next_start_ms','diff_start_ms','grad','cum_sum']])

			extended_music_dataframe_list.append(df_gb
			 .groupby(['cum_sum','pitch','velocity','part'])
			 .agg({'start_ms':min, 'dur_ms':sum})
			 .reset_index()
			 [['dur_ms','velocity','pitch','part','start_ms']]
			 )

		print('-------Fallan aqui los .loc --- 2')
		#print(pd.concat(extended_music_dataframe_list).sort_values(['part','start_ms']))

		return pd.concat(extended_music_dataframe_list).sort_values(['part','start_ms'])



class SequenceChordPolyphony (Polyphony):

	def __init__(self, sequenceChord):
		self.sequenceChord = sequenceChord