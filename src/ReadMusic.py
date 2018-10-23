#!/usr/bin/env python
# -*- coding: utf-8 -*-

from MusicalNote import *
from melody import *
from Polyphony import *
from Play import CSVtoMIDI
from Tonality import *

import numpy as np
import pandas as pd

from collections import Counter, OrderedDict
import itertools

import logging
FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel('INFO')

import sys

import os
import urllib2

# import folders with the code
# At the moment, there is not other way of importing 

# Importing files from c-rnn-gan to read music
# sys.path.insert(0, '/Users/adesant3/Documents/Kindergarten/chopyn/c-rnn-gan')
# from music_data_utils import *



class Score(object):

	pass


class Read(Score):

	def __init__(self, name_file_midi):

		logger.info('INFO: %s', 'Creating class '+self.__class__.__name__)

		self.map_tonic_with_scale = \
					 ({'Do':['Do','Re','Mi','Fa','Sol','La','Si'],
		               'Re':['Re','Mi','Fa#','Sol','La','Si','Do#'],
		               'Reb':['Reb','Mib','Fa','Solb','Lab','Sib','Do'],
		               'Mi':['Mi','Fa#','Sol#','La','Si','Do#','Re#'],
		               'Mib':['Mib','Fa','Sol','Lab','Sib','Do','Re'],
		               'Fa':['Fa','Sol','La','Sib','Do','Re','Mi'],
		               # Fa# is the same as Solb
		               'Sol':['Sol','La','Si','Do','Re','Mi','Fa#'],
		               'Solb':['Solb','Lab','Sib','Dob','Reb','Mib','Fa'],
		               'La':['La','Si','Do#','Re','Mi','Fa#','Sol#'],
		               'Lab':['Lab','Sib','Do','Reb','Mib','Fa','Sol'],
		               'Si':['Si','Do#','Re#','Mi','Fa#','Sol#','La#'],
		               'Sib':['Sib','Do','Re','Mib','Fa','Sol','La'],
		              })

		self.map_note_with_alias = \
						{'Do#':'Reb', 'Reb':'Do#',
						 'Re#':'Mib', 'Mib':'Re#',
						 'Fa':'Mi#', 'Mi#':'Fa',
						 'Fa#':'Solb', 'Solb':'Fa#',
						 'Sol#':'Lab',  'Lab':'Sol#',
						 'La#':'Sib',  'Sib':'La#',
						 'Si#':'Do',  'Do':'Si#',
						}

		self.map_grades_with_scale_position = \
						{'I':0,
						 'II':1,
						 'III':2,
						 'IV':3,
						 'V':4,
						 'VI':5,
						 'VII':6
						}

		self.grades = ['I','II','III','IV','V','VI','VII']


		# Read midi file
		self.music_df = pd.read_csv(name_file_midi)
		# Calculate attribute name_note
		self.calculate_name_note()
		self.divide_music_with_most_granular_tick()

	def get_map_tonic_with_scale(self):
		return self.map_tonic_with_scale

	def get_map_note_with_alias(self):
		return self.map_note_with_alias

	def get_map_grades_with_scale_position(self):
		return self.map_grades_with_scale_position

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

		# Names can be F3#, A2#, then return F# and A#
		# Names can be F3, A2, then return F and A
		self.music_df['octave_name_note'] = \
		(self.music_df
		 .apply(lambda x:
		        _get_note_name_with_octave(x['fullNoteOctave']), axis=1)
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

	def get_tonality(self):
		tonalities = \
					 ({'Do':['Do','Re','Mi','Fa','Sol','La','Si'],
		               'Re':['Do#','Re','Mi','Fa#','Sol','La','Si'],
		               #'Reb':['Do','Reb','Mib','Fa','Solb','Lab','Sib'],
		               'Reb':['Do','Do#','Re#','Fa','Fa#','Sol#','La#'],
		               'Mi':['Do#','Re#','Mi','Fa#','Sol#','La','Si'],
		               #'Mib':['Do','Re','Mib','Fa','Sol','Lab','Sib'],
		               'Mib':['Do','Re','Re#','Fa','Sol','Sol#','La#'],
		               'Fa':['Do','Re','Mi','Fa','Sol','La','La#'],
		               # Fa# is the same as Solb
		               #'Fa#':['Do#','Re#','Fa','Fa#','Sol#','La#','Si'],
		               'Sol':['Do','Re','Mi','Fa#','Sol','La','Si'],
		               #'Solb':['Dob','Reb','Mib','Fa','Solb','Lab','Sib'],
		               'Solb':['Si','Do#','Re#','Fa','Fa#','Sol#','La#'],
		               'La':['Do#','Re','Mi','Fa#','Sol#','La','Si'],
		               #'Lab':['Do','Reb','Mib','Fa','Sol','Lab','Sib'],
		               'Lab':['Do','Do#','Re#','Fa','Sol','Sol#','La#'],
		               'Si':['Do#','Re#','Mi','Fa#','Sol#','La#','Si'],
		               #'Sib':['Do','Re','Mib','Fa','Sol','La','Sib'],
		               'Sib':['Do','Re','Re#','Fa','Sol','La','La#'],
		              })

		note_histogram = \
			(self
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
		      #.head(10)['name_note'][0:10]
		      ).set_index('name_note')['cum_duration'].to_dict()

		tonality_candidates = {}
		for tonality_name, tonality_scale in tonalities.iteritems():
			tonality_candidates[tonality_name] = np.sum([note_histogram.get(iter_tonality_scale,0) 
			                                            for iter_tonality_scale in tonality_scale])

		return next(iter(OrderedDict(sorted(tonality_candidates.items(), key=lambda t: -t[1]))))	

	def apply_tonality(self):

		map_tonic_with_scale = self.get_map_tonic_with_scale()
		
		map_note_with_alias  = self.get_map_note_with_alias()

		

		# Return the aggregates of the chord and their sequence
		agg_criteria = 'octave_name_note'
		#agg_criteria = 'name_note'
		chord_df = self.aggregate_chord_from_tick(aggregation_criteria = agg_criteria)
		
		all_notes = list(np.unique(list(itertools.chain(*chord_df[agg_criteria]))))

		tonic = self.get_tonality()



		# Find the intersection between notes in the scale and notes in the piece of music
		tonic_scale_notes = map_tonic_with_scale[tonic]
		if agg_criteria == 'octave_name_note':
			# In the case, there are, for instance, values such as F4# or G5#
			common_notes = list(set(tonic_scale_notes) & set([iter_notas[:-1] for iter_notas in all_notes]))
		else:
			common_notes = list(set(tonic_scale_notes) & set(all_notes))

		missing_notes_in_scale = list(set(tonic_scale_notes) - set(common_notes))

		# Convert notes in music to the closest in tonic scale
		renamed_missing_notes = \
							([map_note_with_alias[renamed_notes] 
		                     for renamed_notes in missing_notes_in_scale 
		                     if renamed_notes in map_note_with_alias.keys()]
		                     )

		print(renamed_missing_notes)


		# Apply the mapping transformation to the remaining notes
		if agg_criteria == 'octave_name_note':
			chord_df['chord'] = \
			(chord_df[agg_criteria]
			 .apply(lambda tuple_x: 
			        tuple([map_note_with_alias[renamed_notes[:-1]]+renamed_notes[-1]
			              if renamed_notes[:-1] in renamed_missing_notes 
			              else renamed_notes
			              for renamed_notes in tuple_x
			              ])))
		else:
			chord_df['chord'] = \
			(chord_df[agg_criteria]
			 .apply(lambda tuple_x: 
			        tuple([map_note_with_alias[renamed_notes] 
			              if renamed_notes in renamed_missing_notes 
			              else renamed_notes
			              for renamed_notes in tuple_x
			              ])))

		# Convert chord into grades
		if agg_criteria == 'octave_name_note':
			chord_df['grades'] = \
			(chord_df['chord']
			 .apply(lambda tuple_x:
			        tuple([self.grades[tonic_scale_notes
			              .index(chord_element[:-1])]+chord_element[-1]
			              if chord_element[:-1] in tonic_scale_notes 
			              else self._apply_tonality_to_altered_notes(chord_element, tonic_scale_notes)
			              # else 'X'
			              for chord_element in tuple_x
			              ])))
		else:
			chord_df['grades'] = \
			(chord_df['chord']
			 .apply(lambda tuple_x:
			        tuple([self.grades[tonic_scale_notes
			              .index(chord_element)]
			              if chord_element in tonic_scale_notes 
			              else 'X'
			              for chord_element in tuple_x
			              ])))

		chord_df['dur'] = chord_df['max_tick']-chord_df['min_tick']+20


		return chord_df[['chord','grades','dur']]

	def _apply_tonality_to_altered_notes(self, chord_element, tonic_scale_notes):
		
		# position_in_tonic_scale is an array of one element, i.e. [2]
		# If chord_element[:-1] is, for instance, Si, and the scale is Reb
		# then, position_in_tonic_scale will contain [6]

		# It attempts at finding which note within the scale is called in the same way
		position_in_tonic_scale = [position for position, scale_note in enumerate(tonic_scale_notes) 
								   # Sol and Sol# (compare if they coincide in more than one)
								   # Avoid Si and Sol# which will give you one coincidence
								   if scale_note[:2] == chord_element[:2]][0]

		# In case, there are flats, and the note is called similarly, 
		# Then the grade must be the same than the corresponding note in the scale but with +
		# Sib (IV) -> Si natural -> (IV+)
		if tonic_scale_notes[position_in_tonic_scale][-1] == 'b':
			chord = self.grades[position_in_tonic_scale]

			# If the note is Si and Dob is within the scale (only happens in Solb)
			# Then they are the same note and should be corrected.
			if ('Dob' in tonic_scale_notes) & (chord_element[:-1] == 'Si'): 
				chord = self.grades[tonic_scale_notes.index('Dob')]

		elif tonic_scale_notes[position_in_tonic_scale][-1] == '#':
			chord = self.grades[(position_in_tonic_scale-1) % 7]

		else:
			chord = self.grades[position_in_tonic_scale]

		return chord+chord_element[-1]+'+'

	def divide_music_with_most_granular_tick(self):

		# Obtain the histograms of ticks
		# Counter({80.0: 946, 0.0: 642, 40.0: 15, 240.0: 8, 60.0: 6, 20.0: 3, 480.0: 2, 120.0: 2, nan: 1, 160.0: 1, 300.0: 1, 360.0: 1})
		hist_ticks = Counter(self.music_df['start_ticks'].diff())
		# Take the minimum_tick different from 0
		# [  0.  20.  40.  60.  80. 120. 160. 240. 300. 360. 480.  nan]
		# In this case, it is 20.
		minimum_tick = int([ticks for ticks in np.sort(hist_ticks.keys()) if ticks > 0][0])

		self.minimum_tick = minimum_tick

		# Divide music with minimum_tick

		self.music_df['num_minimum_ticks'] = \
		((self.music_df['dur_ticks']+1) / minimum_tick)

		granular_music_list = list()

		for index, iter_note in self.music_df.iterrows():
			for iter_num_minimum_ticks in range(int(iter_note['num_minimum_ticks'])):
				granular_music_list.append(([iter_note['start_ticks']+(iter_num_minimum_ticks)*minimum_tick,
				                          iter_note['pitch'],iter_note['velocity'],
				                          iter_note['part'],iter_note['fullNoteOctave'],
				                          iter_note['name_note'],iter_note['octave_name_note']]
				                          ))

		self.granular_music_df = (pd.DataFrame(granular_music_list,
		                                  columns = ['start_ticks','pitch','velocity','part',
		                                  'fullNoteOctave','name_note','octave_name_note']))

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
		           # 'name_note':lambda x: tuple(dict(Counter(x)).keys())}
		           'name_note':lambda x: Counter(x),
		           'octave_name_note':lambda x: Counter(x),
		           }
		           )
		      .reset_index()
		      )
		# Do not forget the aggregated time pero granular chord.

	def aggregate_chord_from_tick(self, aggregation_criteria = 'name_note', dataframe = None):

		# The other option for aggregation_criteria is:
		# aggregation_criteria = 'fullNoteOctave'

		# Given a sequence of notes such as:
		# {u'F2#': 1, u'F1#': 1, u'F3#': 1}
		# {u'F2#': 1, u'F1#': 1, u'F3#': 1}
		# {u'F2#': 1, u'F1#': 1, u'F3#': 1}
		# {u'F2#': 1, u'F1#': 1, u'F3#': 1}
		# {u'F2#': 1, u'F1#': 1, u'F3#': 1}
		# {u'F2#': 1, u'F4#': 1, u'F3#': 1}
		# {u'F2#': 1, u'F4#': 2, u'F3#': 1}

		# The result must be
		# {u'F2#': 5, u'F1#': 5, u'F3#': 5}
		# {u'F2#': 2, u'F4#': 3, u'F3#': 2}

		# First of all calculate chord per ticks
		chord_per_ticks = self.get_chord_from_tick()

		# See the changes in chord per ticks
		changes_in_chords = chord_per_ticks[aggregation_criteria].diff()
		# Since the first element is Nan, force it to be {}
		# It is {} since we are working with dicts
		changes_in_chords[0] = {}

		# Store the final column into id_aggregation_criteria
		chord_per_ticks['id_aggregation_criteria'] = np.cumsum([len(element)>0 
		                                                       for element in changes_in_chords])



		aggregated_chord_per_ticks = (chord_per_ticks
		      .groupby(['id_aggregation_criteria',
		               chord_per_ticks[aggregation_criteria].map(tuple)])
		      .agg({'start_ticks':['min','max']
		           }
		           )
		      .reset_index()
		      )

		# Rename the columns
		aggregated_chord_per_ticks.columns = ['_'.join(col) for col in aggregated_chord_per_ticks.columns]

		# Number of notes of the chord according to aggregation_criteria
		aggregated_chord_per_ticks['num_elements_chord'] = \
		aggregated_chord_per_ticks[aggregation_criteria+'_'].apply(len)

		# Length in time (ticks) of the chord
		aggregated_chord_per_ticks['time_length_chord'] = \
		aggregated_chord_per_ticks['start_ticks_max']-aggregated_chord_per_ticks['start_ticks_min']+self.minimum_tick


		# Analyze those notes that are not individual notes
		tonic_chord_candidates = \
		(aggregated_chord_per_ticks
		      #.loc[aggregated_chord_per_ticks['num_elements_chord']==4]
		      .groupby(aggregation_criteria+'_')
		      .agg({'time_length_chord':'sum'})
		      .filter([aggregation_criteria+'_','time_length_chord'])
		      .sort_values('time_length_chord',ascending=False)
		      .reset_index()
		      )
		
		tonic_chord_candidates['n_elmnts_chord'] = tonic_chord_candidates[aggregation_criteria+'_'].apply(len)

		tonic_chord_candidates.columns = ['notes','ticks','num_el']

		aggregated_chord_per_ticks.columns = \
		['seq_id', aggregation_criteria, 'min_tick','max_tick', 'len', 'time']

		return aggregated_chord_per_ticks

	def convert_grades_sequence_to_notes(self,grades_sequence, tonic):

		tonality_scale = self.get_map_tonic_with_scale()[tonic]
		grade_mapping = self.get_map_grades_with_scale_position()

		notes_sequence = list()

		for chord in grades_sequence:
			notes_chord = list()
			for note in chord:
				if note != 'X':
					# Extract octave
					if note[-1] != '+':
						octave = int(note[-1])
						note_name = note[:-1]
						converted_note = tonality_scale[grade_mapping[note_name]]
						if converted_note[-1] == 'b' or converted_note[-1] == '#':
							alteration = converted_note[-1]
							name_note  = converted_note[:-1]
						else:
							alteration = ''
							name_note  = converted_note

					else:
						# Here, we have VI4+, II5+, ...
						octave = int(note[-2])
						note_name = note[:-2]
						converted_note = tonality_scale[grade_mapping[note_name]]

						if converted_note[-1] == 'b':
							alteration = ''
							name_note  = converted_note[:-1]
						else:
							alteration = '#'
							name_note  = converted_note

						print([octave, note_name, name_note])
				else:
					# In case of X, then use tonic
					converted_note = tonic
					octave = 4
					if tonic[-1] == 'b' or tonic[-1] == '#':
						alteration = tonic[-1]
						name_note  = tonic[:-1]
					else:
						alteration = ''
						name_note  = tonic

				notes_props = {'duration':200, 'intensity':70, 'timbre':1,
						   'alteration':alteration, 'octave':octave}		
				notes_chord.append(globals()[name_note](**notes_props))

			notes_sequence.append(notes_chord)

			# notes_props = {'duration':200, 'intensity':70, 'timbre':1,
			# 			   'alteration':alteration, 'octave':octave}

			# notes_sequence.append(globals()[name_note](**notes_props))
				
		return notes_sequence

	def download_midi_music(self):
		sources                              = {}
		sources['classical']                 = {}
		sources['classical']['alkan']        = ['http://www.classicalmidi.co.uk/alkan.htm']
		sources['classical']['adam']         = ['http://www.classicalmidi.co.uk/adam.htm']
		sources['classical']['aguado']       = ['http://www.classicalmidi.co.uk/aguadodion.htm']	

		midi_files = {}
		datadir = '/Users/adesant3/Documents/Kindergarten/chopyn/data/'


		import urllib2
		response = urllib2.urlopen('http://python.org/')
		html = response.read()


		if os.path.exists(os.path.join(datadir, 'do-not-redownload.txt')):
			print 'Already completely downloaded, delete do-not-redownload.txt to check for files to download.'
			return
		for genre in sources:
			midi_files[genre] = {}
			print(genre)
			for composer in sources[genre]:
				midi_files[genre][composer] = []
	        	for url in sources[genre][composer]:
					print(url)
					response = urllib2.urlopen(url)
					#if 'classicalmidi' in url:
					#  headers = response.info()
					#  print headers
					data = response.read()

					#htmlinks = re.findall('"(  ?[^"]+\.htm)"', data)
					#for link in htmlinks:
					#  print 'http://www.classicalmidi.co.uk/'+strip(link)
	          
					# make urls absolute:
					urlparsed = urlparse.urlparse(url)
					data = re.sub('href="\/', 'href="http://'+urlparsed.hostname+'/', data, flags= re.IGNORECASE)
					data = re.sub('href="(?!http:)', 'href="http://'+urlparsed.hostname+urlparsed.path[:urlparsed.path.rfind('/')]+'/', data, flags= re.IGNORECASE)
					#if 'classicalmidi' in url:
					#  print data
	          
					links = re.findall('"(http://[^"]+\.mid)"', data)
					for link in links:
						cont = False
						for p in ignore_patterns:
							if p in link:
								print 'Not downloading links with {}'.format(p)
								cont = True
								continue
						if cont: continue
						print link
						filename = link.split('/')[-1]
						valid_chars = "-_.()%s%s" % (string.ascii_letters, string.digits)
						filename = ''.join(c for c in filename if c in valid_chars)
						print genre+'/'+composer+'/'+filename
						midi_files[genre][composer].append(filename)
						localdir = os.path.join(os.path.join(self.datadir, genre), composer)
						localpath = os.path.join(localdir, filename)
						if os.path.exists(localpath):
							print 'File exists. Not redownloading: {}'.format(localpath)
						else:
							try:
								response_midi = urllib2.urlopen(link)
								try: os.makedirs(localdir)
								except: pass
								data_midi = response_midi.read()
								if 'DOCTYPE html PUBLIC' in data_midi:
									print 'Seems to have been served an html page instead of a midi file. Continuing with next file.'
								elif 'RIFF' in data_midi[0:9]:
									print 'Seems to have been served an RIFF file instead of a midi file. Continuing with next file.'
								else:
									with open(localpath, 'w') as f:
										f.write(data_midi)
							except:
								print 'Failed to fetch {}'.format(link)
		with open(os.path.join(self.datadir, 'do-not-redownload.txt'), 'w') as f:
			f.write('This directory is considered completely downloaded.')

def _get_note_name_without_octave(fullNoteOctave):
	# Function to get the name, regardless the octave

	notes_dict = {'A':'La','B':'Si','C':'Do','D':'Re','E':'Mi','F':'Fa','G':'Sol'}

	if len(fullNoteOctave) == 3:
		# Names can be F3#, A2#, then return F# and A#
		return notes_dict[fullNoteOctave[0]]+fullNoteOctave[2]
	else:
		# Names can be F3, A2, then return F and A
		return notes_dict[fullNoteOctave[0]]

def _get_note_name_with_octave(fullNoteOctave):
	# Function to get the name INCLUDING the octave

	notes_dict = {'A':'La','B':'Si','C':'Do','D':'Re','E':'Mi','F':'Fa','G':'Sol'}

	if len(fullNoteOctave) == 3:
		# Names can be F3#, A2#, then return F# and A#
		return notes_dict[fullNoteOctave[0]]+fullNoteOctave[2]+fullNoteOctave[1]
	else:
		# Names can be F3, A2, then return F and A
		return notes_dict[fullNoteOctave[0]]+fullNoteOctave[1]

def _convert_note_into_grade(chord_tuple):
	# Function to convert a chord such as (F4#, C5#, F5, A4) into 
	# grade functionalities (II,III,I,IV) for instance.
	# In this way, music is normalized and harmonic structure may be learned.

	tonality = self.get_tonality()

if __name__ == "__main__":

	name_file_midi = '../../scores/Schubert_S560_Schwanengesang_no7.csv'
	name_file_midi = '../../scores/Brahms_symphony_2_2.csv' # Si M
	name_file_midi = '../../scores/Brahms_symphony_2_1.csv'
	name_file_midi = '../../scores/Albeniz_Asturias.csv'
	name_file_midi = '../../scores/Schuber_Impromptu_D_899_No_3.csv'
	name_file_midi = '../../scores/Chopin_Etude_Op_10_n_1.csv'
	name_file_midi = '../../scores/Bach-Partita_No1_in_Bb_BWV825_7Gigue.csv'
	name_file_midi = '../../scores/Mozart_Sonata_16.csv'
	name_file_midi = '../../scores/Chopin_Etude_Op_10_n_5.csv'
	name_file_midi = '../../scores/Bach_Cello_Suite_No_1.csv'
	name_file_midi = '../../scores/Debussy_Claire_de_Lune.csv'
	#name_file_midi = '../../scores/Beethoven_Moonlight_Sonata_third_movement.csv'
	#name_file_midi = '../../scores/Schubert_Piano_Trio_2nd_Movement.csv'
	
	musical_piece = Read(name_file_midi)
	# print(chopin.get_music_data().head())
	#print(chopin.get_chord_from_tick().filter(['fullNoteOctave']))
	# print('La tonalidad es: '+musical_piece.get_tonality())
	# grades_chords = chopin.apply_tonality()
	# grades_chords.to_csv('../tmp/'+name_file_midi[13:-4]+'_grades_chords.csv',
	#                      header=True,
	#                      index_label=None)

	# print(grades_chords)

	grades_chords = musical_piece.download_midi_music()
	print(grades_chords)





