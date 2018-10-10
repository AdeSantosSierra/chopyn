#!/usr/bin/env python
# -*- coding: utf-8 -*-

from MusicalNote import *
from melody import *
from Polyphony import *
from Play import CSVtoMIDI

import numpy as np

# Tonality
# Two types of tonalities: Major and Minor


class Tonality (object):
    # The class tonality defines the structure of the music
    # (2) The relationships between grades and notes
    # (3) Which is the Tonica/Dominant/... function of every note

    def __init__(self, Tonic):
        # Tonic is the main note - This must be set as a parameter

        # Names of the sequences/grades I, II, III, IV, ...
        self.dict_grades_positions = {'I': 0, 'II': 1, 'III': 2, 'IV': 3, 'V': 4, 'VI': 5, 'VII': 6}

        # Which is the Tonic Note
        self.tonic = Tonic 

    def get_tonic(self):
        return self.tonic


class Major (Tonality):
    # (1) Different chords of every functional note
    # (2) How to create music from chords and functional notes (I, II, III, ...)
    def __init__(self, Tonic):
		self.major_chords = ['I', 'IV', 'V']
		self.minor_chords = ['II', 'III', 'VI']
		self.dis_chords   = ['VII']

		super(self.__class__, self).__init__(Tonic)

		

    def create_music_from_grades_sequences(self, 
                                           musical_sequence, 
                                           number_of_notes_per_compass):

        melody_sequence = []

        self.grades = self.get_tonic().get_major_scale()
        print([note.to_string() for note in self.grades])

        for grade in musical_sequence:
            # (1) Iterate every single grade within musical_sequence
            # (2) Extract, for every note, the corresponding chord
            # (3) Play random music ONLY with those notes within the chord
            # (4) Concatenate all the notes
            # main_note_from_grade = (globals()
            #                         [self.grades[self.dict_grades_positions[grade]].__name__]())

            main_note_from_grade = (self.grades[self.dict_grades_positions[grade]])

            # (2) Extract the chord for every note
            if grade in self.major_chords:
                notes_from_chord = main_note_from_grade.get_major_chord()
            else:
                notes_from_chord = main_note_from_grade.get_minor_chord()




            # (3) and (4) Play random music ONLY with notes from the chord
            melody_sequence.extend(np.random.choice(notes_from_chord, 
                                                    number_of_notes_per_compass, 
                                                    replace=True))
            #print(melody_sequence)

        # (5) First and Last note must be the Tonic
        melody_sequence[0] = self.get_tonic()
        melody_sequence[-1] = self.get_tonic()

        # (6) Add properties (duration, intensity, timbre)
        melody_sequence_to_play = []
        for note in melody_sequence:
            note.update_props({'duration': np.random.choice([125, 250], 1, replace=True)[0], 
                               'intensity': 70, 'timbre': 1})
            melody_sequence_to_play.append(note)

        return melody_sequence_to_play


if __name__ == '__main__':
    tonal = Tonality(Sol)

    do = Major(La(**{'alteration':'b'}))
    musical_sequence = ['I', 'V', 'VI', 'III', 'IV','I', 'IV', 'V', 'I']
    number_of_notes_per_compass = 10
    melody_sequence = do.create_music_from_grades_sequences(musical_sequence, 
                                                            number_of_notes_per_compass)

    melody = SequenceMelody(melody_sequence)
    CSVtoMIDI(melody.convert_to_midi())
