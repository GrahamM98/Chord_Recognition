from test_funcs import noteDict, notYetImpl

# returns true if all tokens are valid note names
# otherwise return false			
def areValidNoteNames(noteTokens):
    validNotes = list(noteDict.keys())
    validNotes.sort()

    for i in range(0, len(noteTokens)):
        noteTokens[i] = noteTokens[i].upper()
        # check if note token is valid
        if noteTokens[i] not in noteDict.keys():
            print("error: '%s' is an valid note name" % noteTokens[i])
            print("valid note names: ", end="")
            print(validNotes)
            return False
    return True

def getNoteNames(noteCnt):
    print("Input %d note names, then press enter" % noteCnt)
    while True:
        noteTokens = input("> ").split(" ")
        if len(noteTokens) != noteCnt:
            print("error: invalid number of notes, please enter %d note names" % noteCnt)
            continue
        if areValidNoteNames(noteTokens):
            break
    return noteTokens
def testN2C():
    noteNames = getNoteNames(3)
    notYetImpl()
