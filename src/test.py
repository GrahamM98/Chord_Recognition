from test_dict import testDict
from test_funcs import runHelp


# hang I/O request
# end program when "exit" is entered
def getIO():
    while True:
        # prompt
        print("Choose desired action (type help for more information)")
        userIn = input("> ").lower().rstrip()
        # upon unrecognized command
        if userIn not in list(testDict.keys()) + ["exit", "help"]:
            print("'%s' is not a recognized command. Type help for list of valid commands." % userIn)
        elif userIn == "exit":
            print("bye") 
            exit()
        elif userIn == "help": runHelp(testDict)
        else:
            testDict[userIn].method()
            exit()
