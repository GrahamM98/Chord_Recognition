from test_funcs import noteDict, notYetImpl
import n2cBP
import tensorflow as tf

chordDict = {
        "Major": 0,
        "Minor": 1,
        "Augmented": 2,
        "Diminished": 3,
    }

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
    # setting up inputs and outputs
    n2c_inputs = tf.placeholder(tf.float32, [None, 3])
    n2c_outputs = tf.placeholder(tf.float32, [None, 4])

    # setting up weights and biases for inputs->hiddens
    n2c_weights_in = tf.Variable(tf.random_normal([3, 15], stddev=0.03), name='n2c_weights_in')
    n2c_biases_in = tf.Variable(tf.random_normal([15]), name='n2c_biases_in')

    # setting up weights and biases for hiddens->outputs
    n2c_weights_out = tf.Variable(tf.random_normal([15, 4], stddev=0.03), name='n2c_weights_out')
    n2c_biases_out = tf.Variable(tf.random_normal([4]), name='n2c_biases_out')

    # setting up math for hidden layer output
    n2c_hidden_out = tf.add(tf.matmul(n2c_inputs, n2c_weights_in), n2c_biases_in)
    n2c_hidden_out = tf.nn.relu(n2c_hidden_out)

    # setting up math for outputs
    n2c_output_values = tf.nn.softmax(tf.add(tf.matmul(n2c_hidden_out, n2c_weights_out), n2c_biases_out))

    # setting up math for backpropagation
    n2c_outputs_clipped = tf.clip_by_value(n2c_output_values, 1e-10, 0.9999999)
    n2c_cross_entropy = -tf.reduce_sum(n2c_outputs * tf.log(n2c_outputs_clipped))

    #set up optimizer
    n2c_optimizer = tf.train.GradientDescentOptimizer(0.003)
    n2c_train_step = n2c_optimizer.minimize(n2c_cross_entropy)

    # set up initialisation operator
    n2c_init = tf.global_variables_initializer()

    # function for correct prediction
    n2c_correct_prediction = tf.equal(tf.argmax(n2c_outputs, 1), tf.argmax(n2c_output_values, 1))

    # function for accuracy
    n2c_accuracy = tf.reduce_mean(tf.cast(n2c_correct_prediction, tf.float32))

    #create session
    n2c_sess = tf.Session()

    #initialize session variables
    n2c_sess.run(n2c_init)

    #set up saver to save network
    saver = tf.train.Saver()

    saver.restore(n2c_sess, "checkpoints/n2cBP.ckpt")

    correct_out = [0]*4
    noteNames = getNoteNames(3)
    n2c_notes = [n2cBP.noteDict[noteNames[0]], n2cBP.noteDict[noteNames[1]], n2cBP.noteDict[noteNames[2]]]

    if n2c_notes[1] == n2c_notes[0]+3:
        n2norm = -1
    elif n2c_notes[1] == n2c_notes[0]+4:
        n2norm = 1
    else:
        n2norm = 0

    if n2c_notes[2] == n2c_notes[1]+3:
        n3norm = -1
    elif n2c_notes[2] == n2c_notes[1]+4:
        n3norm = 1
    else:
        n3norm = 0


    n2c_notes = [0, n2norm, n3norm]

    args = n2c_sess.run([n2c_output_values], feed_dict={n2c_inputs: [n2c_notes]})
    out = args[0][0]
    maxIndex = 0
    maxVal = 0
    for i in range(4):
        if out[i] > maxVal:
            maxIndex = i
            maxVal = out[i]
    if maxVal < .5:
        print("The given notes don't make a structured chord.")
    for entry in chordDict.keys():
        if maxIndex == chordDict[entry]:
            chord = entry
    print("The chord made using %s, %s, and %s is %s" % (noteNames[0], noteNames[1], noteNames[2], chord))
