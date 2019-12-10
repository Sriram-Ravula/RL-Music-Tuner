"""
Command:
python one_hot_to_midi.py <input numpy file> <output midi file>
"""

from mido import Message, MidiFile, MidiTrack
import numpy as np
import sys

index_to_pitch_map = {
    0 : 36,
    1 : 38,
    2 : 50,
    3 : 47,
    4 : 43,
    5 : 46,
    6 : 42,
    7 : 49,
    8 : 51,
    9 : 0 # value 0 is not defined on channel 9 so it's just silence (channel 9 is for drums)
}

def index_to_pitch(index):
    return index_to_pitch_map[index]

def one_hot_to_midi(one_hot, midi_filename = 'song.mid'):
  mid = MidiFile()
  track = MidiTrack()
  mid.tracks.append(track)

  pitch = 36
  duration = 128
  velocity = 64

  track.append(Message('program_change', program=12, time=0))

  #for sb in one_hot:
  for i in range(one_hot.shape[1]):
    sb = one_hot[:,i]
  
    pitch = int(index_to_pitch(np.argmax(sb)))
    track.append(Message('note_on', channel=9, note=pitch, velocity=64, time=duration))
    track.append(Message('note_off', channel=9, note=pitch, velocity=127, time=duration))

  mid.save(midi_filename)

np_file = sys.argv[1]
midi_file = sys.argv[2]

one_hot = np.load(np_file)
one_hot_to_midi(one_hot, midi_filename = midi_file)
