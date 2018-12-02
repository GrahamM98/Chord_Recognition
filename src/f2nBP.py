import tensorflow as tf
import random
import aubio

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

#returns the average between two numbers
def avg(v1, v2):
    return (v1+v2)/2

#normalizes an input frequency
def normValue(start, end, freq):
    return (freq-avg(start, end))/(start/2)

#this net will have one input and 12 outputs

def make_frequency_to_note_neural_net():
    # all variables are prefixed with f2n so that the tensorflow session can
    # differentiate between the variables relevant to this neural net and the
    # chord neural net.

    # setting up inputs and outputs
    f2n_inputs = tf.placeholder(tf.float32, [None, 1])
    f2n_outputs = tf.placeholder(tf.float32, [None, 12])

    # setting up weights and biases for inputs->hiddens
    f2n_weights_in = tf.Variable(tf.random_normal([1, 15], stddev=0.03), name='f2n_weights_in')
    f2n_biases_in = tf.Variable(tf.random_normal([15]), name='f2n_biases_in')

    # setting up weights and biases for hiddens->outputs
    f2n_weights_out = tf.Variable(tf.random_normal([15, 12], stddev=0.03), name='f2n_weights_out')
    f2n_biases_out = tf.Variable(tf.random_normal([12]), name='f2n_biases_out')

    # setting up math for hidden layer output
    f2n_hidden_out = tf.add(tf.matmul(f2n_inputs, f2n_weights_in), f2n_biases_in)
    f2n_hidden_out = tf.nn.relu(f2n_hidden_out)

    # setting up math for outputs
    f2n_output_values = tf.nn.softmax(tf.add(tf.matmul(f2n_hidden_out, f2n_weights_out), f2n_biases_out))

    # setting up math for backpropagation
    f2n_outputs_clipped = tf.clip_by_value(f2n_output_values, 1e-10, 0.9999999)
    #f2n_cross_entropy = -tf.reduce_mean(tf.reduce_sum(f2n_outputs * tf.log(f2n_outputs_clipped) + (1 - f2n_outputs) * tf.log(1 - f2n_outputs_clipped), axis=1))
    f2n_cross_entropy = -tf.reduce_sum(f2n_outputs * tf.log(f2n_outputs_clipped))

    #set up optimizer
    f2n_optimizer = tf.train.GradientDescentOptimizer(0.003)
    f2n_train_step = f2n_optimizer.minimize(f2n_cross_entropy)

    # set up initialisation operator
    f2n_init = tf.global_variables_initializer()

    # function for correct prediction
    f2n_correct_prediction = tf.equal(tf.argmax(f2n_outputs, 1), tf.argmax(f2n_output_values, 1))
    
    # function for accuracy
    f2n_accuracy = tf.reduce_mean(tf.cast(f2n_correct_prediction, tf.float32))
    
    #create session
    f2n_sess = tf.Session()

    #initialize session variables
    f2n_sess.run(f2n_init)

    batch_size = 100
    
    #test each octave individually
    for octave in range(0, 6):
        
        print("octave: " + str(octave))
                
        #calculate the start and enc of each octave
        start = 110 * pow(2, octave)
        end = 110 * pow(2, octave+1)
        
        for epoch in range(0, 10000):
                
            #establish list for the batches of inputs and outputs
            f2n_freq = [0]*batch_size
            correct_out = [None]*batch_size
            
            #create batches
            for i in range(batch_size):
                correct_out[i] = [0]*12
                train_freq = random.randrange(start, end)
                f2n_freq[i] = [normValue(start, end, train_freq)]

                #find desired note, continue if frequency is invalid
                try:
                    f2n_desired = aubio.freq2note(train_freq)[:-1]
                except:
                    continue

                try:
                    correct_out[i][noteDict[f2n_desired]-1] = 1
                except:
                    continue

            #find accuracy and loss using session
            a, c, _ = f2n_sess.run([f2n_accuracy, f2n_cross_entropy, f2n_train_step], feed_dict={f2n_inputs: f2n_freq, f2n_outputs: correct_out})

            #print data every 1000th iteration
            if (epoch + 1) % 1000 == 0: 
                #create test data
                f2n_freq_test = [0]*batch_size
                correct_out_test = [None]*batch_size
                for i in range(batch_size):
                    correct_out_test[i] = [0]*12
                    test_freq = random.randrange(start, end)
                    f2n_freq_test[i] = [normValue(start, end, test_freq)]

                    #find desired note, continue if frequency is invalid
                    try:
                        f2n_desired = aubio.freq2note(test_freq)[:-1]
                    except:
                        continue

                    try:
                        correct_out_test[i][noteDict[f2n_desired]-1] = 1
                    except:
                        continue
                a_test, _ = f2n_sess.run([f2n_accuracy, f2n_cross_entropy], feed_dict={f2n_inputs: f2n_freq_test, f2n_outputs: correct_out_test})

                print("Train Data Loss: " + str(c))
                print("Train Data Accuracy: " + str(100.00*a) + "%")
                print("Test Data Accuracy: " + str(100.00*a_test) + "%")
                #print("Test input data: ")
                #for entry in f2n_freq_test:
                #    print(entry)
                print()

    return f2n_sess
