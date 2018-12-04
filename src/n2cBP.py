import random
from itertools import permutations as p

noteDict = {
        "A": 1,
        "A#": 2,
        "B": 3,
        "C": 4,
        "C#": 5,
        "D": 6,
        "D#": 7,
        "E": 8,
        "F": 9,
        "F#": 10,
        "G": 11,
        "G#": 12
    }

def findChord(notes):
    print("\nChord structure: ", end='')
    for entry in p(notes):
        triad = [None]*3
        for i in range(3):
            triad[i] = noteDict[entry[i]]
        if triad[1] < triad[0]:
            triad[1] += 12
        if triad[2] < triad[1] or triad[2] < triad[0]:
            triad[2] += 12 
        if triad[2]-triad[1] == 3 and triad[1]-triad[0] == 4:
            return "Major"
        if triad[2]-triad[1] == 4 and triad[1]-triad[0] == 3:
            return "Minor"
        if triad[2]-triad[1] == 4 and triad[1]-triad[0] == 4:
            return "Augmented"
        if triad[2]-triad[1] == 3 and triad[1]-triad[0] == 3:
            return "Diminished"
        return "None"
