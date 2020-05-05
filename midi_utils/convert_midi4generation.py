# MIDI utils for conversion
# Based on code written by:
# Kento Watanabe, Yuichiroh Matsubayashi, Satoru Fukayama, Masataka Goto, Kentaro Inui and Tomoyasu Nakano
# https://github.com/KentoW/melody-conditioned-lyrics-language-model

# -*- coding: utf-8 -*-
import sys
import mido
import json
import argparse
from collections import defaultdict

# 480: quater rest/note
# 1920: whole rest/note

def get_length(length):
    length = int(length)
    if length <= 40:
        return 0
    elif length <= 80:
        return 60
    elif length <= 180:
        return 120
    elif length <= 300:
        return 240
    elif length <= 420:
        return 360
    elif length <= 600:
        return 480
    elif length <= 840:
        return 720
    elif length <= 1080:
        return 960
    elif length <= 1320:
        return 1200
    elif length <= 1560:
        return 1440
    elif length <= 1800:
        return 1680
    elif length <= 2880:
        return 1920
    elif length <= 4800:
        return 3840
    elif length <= 4560:
        return 5760
    else:
        return 7680

def miditix_2_notelenghts(length):
    length = int(length)
    if length <= 40:
        return 0
    elif length <= 80:
        return .25
    elif length <= 180:
        return .25
    elif length <= 300:
        return .5
    elif length <= 420:
        return .75
    elif length <= 600:
        return 1.0
    elif length <= 840:
        return 1.0
    elif length <= 1080:
        return 1.5
    elif length <= 1320:
        return 2.0
    elif length <= 1560:
        return 2.0
    elif length <= 1800:
        return 3.0
    elif length <= 2880:
        return 4.0
    elif length <= 4800:
        return 8.0
    elif length <= 4560:
        return 16.0
    else:
        return 32.0

def convert(midi_file):
    mid = mido.MidiFile(midi_file, ticks_per_beat=480)
    on_queue = defaultdict(list)
    current_position = 0
    notes = []
    for i, track in enumerate(mid.tracks):
        if i == 1:
            for msg in track:
                if msg.type == "note_on":
                    note = msg.note
                    length = get_length(msg.time)
                    current_position += length
                    on_queue[note].append(current_position)


                elif msg.type == "note_off":
                    note = msg.note
                    length = get_length(msg.time)
                    current_position += length

                    if len(on_queue[note]) > 0:
                        start_position = on_queue[note][0]
                        duration = current_position - start_position
                        notes.append((note, start_position, duration))
                        on_queue[note].pop(0)

    fsp = notes[0][1]
    prev_end_position = 0
    notes4generation = [("rest", "32.0")]
    for note in notes:
        note_number = note[0]
        start_position = note[1] - fsp
        duration = note[2]
        if start_position < prev_end_position:
            continue
        elif start_position == prev_end_position:
            _temp_duration = get_length(duration)
            if _temp_duration > 0:
                new_note_dur = miditix_2_notelenghts(_temp_duration)
                notes4generation.append((str(note_number), str(new_note_dur)))
        else:
            rest_duration = get_length(start_position - prev_end_position)
            rest_start_position = prev_end_position
            new_rest_dur = miditix_2_notelenghts(rest_duration)
            notes4generation.append(("rest", str(new_rest_dur)))
            _temp_duration = get_length(duration)
            if _temp_duration > 0:
                new_note_dur = miditix_2_notelenghts(_temp_duration)
                notes4generation.append((str(note_number), str(new_note_dur)))
        prev_end_position = start_position + duration
    notes4generation.append(("rest", "32.0"))
    print(notes4generation)
    return notes4generation



def main(args):
    notes = convert(args.midi)
    # print(json.dumps({"melody":notes}))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-midi", "--midi", dest="midi", default="./c-LSTM-LM/sample_data/sample.midi", type=str, help="MIDI file")
    args = parser.parse_args()
    main(args)
