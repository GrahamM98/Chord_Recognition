import tensorflow as tf

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
    f2n_cross_entropy = -tf.reduce_mean(tf.reduce_sum(f2n_outputs * tf.log(f2n_outputs_clipped) + (1 - f2n_outputs) * tf.log(1 - f2n_outputs_clipped), axis=1))

    # set up initialisation operator
    f2n_init = tf.global_variables_initializer()

    # function for correct prediction
    f2n_correct_prediction = tf.equal(tf.argmax(f2n_outputs, 1), tf.argmax(f2n_output_values, 1))
    
    # function for accuracy
    f2n_accuracy = tf.reduce_mean(tf.cast(f2n_correct_prediction, tf.float32))

    return tf.Session()
