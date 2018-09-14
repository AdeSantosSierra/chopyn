#!/usr/bin/env python
# -*- coding: utf-8 -*-

class CSVtoMIDI(object):

	# Class to convert CSV into MIDI
	# Add DOC later
    

	def __init__(self,dataframe):
		
		
		# Create the names for the csv and midi files
		name_file_midi = 'my_first_midi_file'
		csv_file = '../output/CSV/'+name_file_midi+'.csv'
		midi_file = '../output/MIDI/'+name_file_midi+'.midi'

		# Store dataframe
		self.dataframe = dataframe

		self.create_csv(csv_file)
		self.convert_csv_to_midi(csv_file,midi_file)

	def init_dict (self,all_the_voices):
		# Init the dictionary which will collect all the lines from the MIDI file
		# all_the_voices contains all the instruments/voices of the composition
		# The idea is to create a dictionary with the header of every voice, which must include
		# Start_track and Program_c
		self.dic = {key: [str(key+1)+', 0, Start_track',
						  # 0, Program_c, Channel, Instrument
                          str(key+1)+', 0, Program_c, 1, 1'] for key in all_the_voices}


	def appendNote (self,row):
		# For every single note and voice/instrument, this method will include one row per note, 
		# for the corresponding voice. Sometimes, voice/instrument is called part.

		# Example:
		# 3, 0, Note_on_c, 3, 62, 28
		# 3, 250, Note_off_c, 3, 62, 0
		# 3, 250, Note_on_c, 3, 53, 34
		# 3, 375, Note_off_c, 3, 53, 0
		# 3, 375, Note_on_c, 3, 62, 30
		# 3, 500, Note_off_c, 3, 62, 0

        # Note_on_c
		part = row['part']
		self.dic[part].append(', '.join([str(row['part']+1),
					str(row['start_ms']),
					'Note_on_c',
					str(row['part']+1), # Channel
					str(row['pitch']),
					str(row['velocity'])
					]))

        # Note_off_c
		self.dic[part].append(', '.join([str(row['part']+1),
                  (str(row['start_ms']+row['dur_ms'])),
                  'Note_off_c',
                  str(row['part']+1), # Channel
                  str(row['pitch']),
                  '0'
                 ]))
		

	def create_csv(self, csv_file):
    	# Create CSV file

		csv = open(csv_file, 'w')
		self.init_dict(self.dataframe.part.unique())

		# Number of voices/instruments should be given by datamart_to_save.part.max()
		header_list = \
		['0, 0, Header, 1, '+str(self.dataframe.part.max()+1)+', 480',
		# Not needed now
		# '1, 0, Start_track',
		# '1, 0, Title_t, "Close Encounters"',
		# '1, 0, Text_t, "Sample for MIDIcsv Distribution"',
		# '1, 0, Copyright_t, "This file is in the public domain"',
		# '1, 0, Time_signature, 4, 2, 24, 8',
		# '1, 0, Tempo, 500000',
		# '1, 0, End_track',
		'']
		
		# Write the header into the CSV file
		csv.write('\n'.join(header_list))

		# For every row in the dataframe, apply the method appendNote, which will store in self.dic
		# one line per note
		self.dataframe.apply(self.appendNote,axis=1)

		# Add the last line per voice: 
		# Example:
		# 2, 83125, End_track

		for iterDic in self.dic.keys():
		    v = self.dic[iterDic][-1].split(',')[:2]
		    v.append(' End_track\n')
		    self.dic[iterDic].append(','.join(v))

		# Save all the notes per voice/instrument in the CSV file.
		# Notice, that every voice will contain:
		# Start_track
		# Notes
		# End_track
		# Example:
		[csv.write('\n'.join(self.dic[iterDict])) for iterDict in self.dic.keys()]

		# End of file
		csv.write('0, 0, End_of_file')

		# Close the file
		csv.close()

	def convert_csv_to_midi(self,csv_file,midi_file):
		# Convert CSV to MIDI
		# It is needed to use the csvmidi command
		import os
		os.system(' '.join(['csvmidi',csv_file,midi_file]))


if __name__== "__main__":
	import pandas as pd
	print('hola')	
	datamart = pd.read_csv('../../Data Beers/data/NetPerform/generated/Chat_and_SocialNet.csv')
	CSVtoMIDI(datamart)





