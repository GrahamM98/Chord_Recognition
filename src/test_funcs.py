import re
MIN_VAL = 110
MAX_VAL = 14080
#from f2nBP import noteDict
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
# TODO connect to F2N net
# prompts for user to enter an input
# for F2N network to predict a note
def testF2N():
	prompt = "enter integer value between %d and %d " % (MIN_VAL, MAX_VAL)
	intInput = getNetworkVal(prompt)
	notYetImpl()
	# give note predicted
# TODO connect to F2C	
# prompts for 3 inptus 
# for F2C network to predict a chord
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
