import random
import tensorflow as tf
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

chordDict = {
        "Major": 0,
        "Minor": 1,
        "Augmented": 2,
        "Diminished": 3,
    }

def findNote(nNum):
    for entry in noteDict.keys():
        if nNum == noteDict[entry]:
            return entry
    print("%d is an invalid input" % (nNum))

def avg(v1, v2):
    return (v1+v2)/2

def normValue(start, end, val):
    return (val-avg(start, end))/(start/2)


def findChord(notes):
    for entry in p(notes):
        triad = [None]*3
        for i in range(3):
            triad[i] = entry[i]
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

def note_to_chord_neural_net():
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

    batch_size = 100

    for epoch in range(10000):
        n2c_notes = [None]*batch_size
        n2c_correct_out = [None]*batch_size
        train_notes = [None]*batch_size

        for i in range(batch_size):
            train_notes[i] = [0]*3
            n2c_correct_out[i] = [0]*4

            note1 = random.randint(1, 4)
            note2 = random.randint(note1+3, note1+4)
            note3 = random.randint(note2+3, note2+4)

            train_notes[i] = [note1, note2, note3]

            if note2 == note1+3:
                n2norm = -1
            else:
                n2norm = 1

            if note3 == note2+3:
                n3norm = -1
            else:
                n3norm = 1

            n2c_notes[i] = [0, n2norm, n3norm]

            n2c_desired = findChord(train_notes[i])

            if n2c_desired != "None":
                n2c_correct_out[i][chordDict[n2c_desired]] = 1

        a, c, _ = n2c_sess.run([n2c_accuracy, n2c_cross_entropy, n2c_train_step], feed_dict={n2c_inputs:n2c_notes, n2c_outputs: n2c_correct_out})

        if (epoch + 1) % 1000 == 0:
            print("Train data loss: " + str(c))
            print("Train data accuracy: " + str(a))
            print()
    save_path = saver.save(n2c_sess, "checkpoints/n2cBP.ckpt")
    return n2c_sess























