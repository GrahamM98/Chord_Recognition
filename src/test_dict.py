from freq2Note import trainF2N
from notes2Chord import trainN2C
from test_funcs import testF2N, testF2C
from testn2c import testN2C
# option descriptions
n2c_description = "enter 3 input notes into n2c network to predict a chord"
f2c_description = "enter 3 integer inputs into f2n & n2c network to predict a chord"	
f2n_description = "enter 1 integer input into f2n network to predict a note"

# contains description and method for each menu option 
class TestObject:
	def __init__(self,inDescription, inMethod):
		self.description = inDescription	
		self.method = inMethod


# key: option name
# value: object containing the option's description and method
testDict = {
	"trainf2n": TestObject("train the f2n neural net", trainF2N),
	"trainf2c": TestObject("train the n2c neural net", trainN2C),
	"testf2n": TestObject(f2n_description, testF2N),
	"testn2c": TestObject(n2c_description,testN2C),
	"testf2c": TestObject(f2c_description, testF2C),
}

