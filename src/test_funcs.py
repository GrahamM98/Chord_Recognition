import tensorflow as tf
import aubio
import re
import f2nBP
MIN_VAL = 110
MAX_VAL = 14080
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

def notYetImpl():
    print("not yet implemented, choose another option")


# prompts user to input an integer between minVal and maxVal
# and returns it	
def getNetworkVal(prompt):
    netVal = MIN_VAL - 1
    nonDigit = re.compile('\D') #regex matches nondigits
    # prompt user for integer in range 
    while True:
        netVal = str(input(prompt))
        # variable type check
        if nonDigit.match(netVal):
            prompt = "invalid variable type. try again "
            netVal = int(MIN_VAL - 1)
            continue
        netVal = int(netVal)
        # range check
        if netVal < MIN_VAL or netVal > MAX_VAL:
            print("out of range. try again")
        else:
            return netVal        

def testF2N():

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

    #set up saver to save network
    saver = tf.train.Saver()

    prompt = "enter integer value between %d and %d " % (MIN_VAL, MAX_VAL)
    intInput = getNetworkVal(prompt)
    #notYetImpl()
    # give note predicted
    
    saver.restore(f2n_sess, "checkpoints/f2nBP.ckpt")
    print("session restored")

    for i in range(0, 7):
        if 110*pow(2, i) <= intInput <= 110*pow(2, i+1):
            start = 110*pow(2, i)
            end = 110*pow(2, i+1)

    correct_out = [0]*12
    f2n_freq = [f2nBP.normValue(start, end, intInput)]
    f2n_desired = aubio.freq2note(intInput)[:-1]
    args = f2n_sess.run([f2n_output_values], feed_dict={f2n_inputs: [f2n_freq]})
    maxEntry = 0
    maxVal = 0
    out = args[0][0]
    for i in range(len(out)):
        if out[i] > maxVal:
            maxVal = out[i]
            maxEntry = i

    for entry in noteDict.keys():
        if noteDict[entry] == maxEntry+1:
            print("Predicted note for frequency %d is %s" % (intInput, entry))

def testF2C():
    inputCnt = 3
    intInputs = []
    print("enter %d integer(s) in range %d-%d" % (inputCnt, MIN_VAL, MAX_VAL))
    for i in range(1, inputCnt+1):
        networkVal = getNetworkVal("input %d >" % i)
        intInputs.append(networkVal)
    notYetImpl()
    # give chord predicted


# prints all valid commands along with their description
def runHelp(testDict):
    helpDescription = "print all valid commands along with their description"
	
    # sort options alphabetically
    keys = list(testDict.keys())
    keys.sort()
	
    print("Valid commands: ")
    for key in keys:	
        print("\t %-15s %s" % (key, testDict[key].description))	
	
    # print exit and help options
    print("\t %-15s %s" % ("exit", "exit the program"))
    print("\t %-15s %s" % ("help", helpDescription))
