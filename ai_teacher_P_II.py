# Program Name: ai teacher P II
# Name: Charles
# Date: May 13
# Description: This program can produce an AI teacher named P_II, and let it teach user knowledge
#                       about CS and other things.

####################################################################
############################### Imports ###############################
####################################################################
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import trainDataP_II as tdp

####################################################################
########################### Machine Learning ############################
####################################################################

#-----------------------------------------------------------------------------------------------#
################# Texts Preprocess Functions ####################
#-----------------------------------------------------------------------------------------------#
tok = Tokenizer()
# textToNum
# @@para: text
# @@return: tupple
# Description: a function that converts a text in a list into a list of sequences
def textToNum(text):
    tok.fit_on_texts(text)
    wordInd = tok.word_index
    # number of words plus one empty word as place holder for padding
    numWords = len(wordInd) + 1
    #convert to sequence
    trainSeq = []
    for i in text:
            token_list = tok.texts_to_sequences([i])[0]
            for j in range(1, len(token_list)):
                    n_gram_sequence = token_list[:j+1]
                    trainSeq.append(n_gram_sequence)
    return wordInd, trainSeq, numWords
# seqToPadded
# @@para: sequence, maxLen
# @@return: array
# Description: a function that converts a list of sequences into a padded list of sequences so that the lengths of sequences are the same
def seqToPadded(sequence, maxLen):
    # pad sequences
    inputPadded = pad_sequences(sequence, maxlen=maxLen, padding='pre')
    return inputPadded

# convertLength
# @@para: sent, size
# @@return: list
# Description: a function that converts a string into a list of strings, each the same given length
def convertLength(sent, size):
    sentList = sent.split()
    lenSent = len(sentList)
    finalSent = []
    numParts = lenSent // size
    if lenSent // size < lenSent / size:
        numParts += 1
    # if the length of input list is larger than the length of default list, cut the input list
    if lenSent > size:
        for i in range(numParts):
            finalSent.append("")
            for j in range(size):
                if i * size + j < lenSent:
                    finalSent[i] += " " + sentList[i * size + j]
    else:
        finalSent = [sent]
    return finalSent
        
    
#-----------------------------------------------------------------------------------------------#
################# Preprocess Text for Training ####################
#-----------------------------------------------------------------------------------------------#
### Original training text, already in the same length
trainSent = tdp.trainDataP_II()

### Convert Text to a List of Sequences
wordInd, trainSeq, numWords= textToNum(trainSent)

### Convert List of Sequences to List of List of Sequences, then pad them
maxLen = 0
for i in range(len(trainSeq)):
    if len(trainSeq[i]) + 1 > maxLen:
        maxLen = len(trainSeq[i]) + 1
inputPad = np.array(seqToPadded(trainSeq, maxLen))

### Separate into x and y (previous words and next word)
x = inputPad[:,:-1]
label = inputPad[:,-1]
y = tf.keras.utils.to_categorical(label, num_classes=numWords)

#-----------------------------------------------------------------------------------------------#
###################### Neural Network ######################
#-----------------------------------------------------------------------------------------------#
model = tf.keras.Sequential([
    # value of each word check
    tf.keras.layers.Embedding(numWords, 240, input_length=maxLen - 1),
    # LSTM layers
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)),
    # dense layer for output
    tf.keras.layers.Dense(numWords, activation='softmax')
])

# loss function & optimizer
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# Train network 100 times with x and y
model.fit(x, y, epochs=30, verbose=1)

####################################################################
########################### P_II & Main Program ##########################
####################################################################

#-----------------------------------------------------------------------------------------------#
########################## P_II ##########################
#-----------------------------------------------------------------------------------------------#
u1Know = "Computer science is the study of computation, automation, and information."
u2Know = "Object-oriented programming is a programming paradigm based on the concept of 'objects', which can contain data and code: data in the form of fields, and code, in the form of procedures."
u3Know = "A UML diagram is a diagram based on the UML (Unified Modeling Language) with the purpose of visually representing a system along with its main actors, roles, actions, artifacts or classes, in order to better understand, alter, maintain, or document information about the system."
u4Know = "Recursion means 'defining a problem in terms of itself'."
u5Know = "In computer science, the time complexity is the computational complexity that describes the amount of computer time it takes to run an algorithm."
u6Know = "Data Structures are a specialized means of organizing and storing data in computers in such a way that we can perform operations on the stored data more efficiently."
engKnow = "English is a language. I am speaking it right now."
hidden = "Philosophy of Applied Mathematical Cognitive Biomechatronics and Foundations of Financial Aerospacial Agricultural Engineering is a new interdisiplinary subject about the greatest study of physical sciences, life sciences, social sciences, humanities, and engineering. It is a new emerging field that can change the world."
class AITeacher:
    # __init__
    # @@para: none
    # @@return: void
    # Description: an initializer for the class AITeacher
    def __init__(self):
        self.name = "P_II"
        self.age = "---"
        self.teachingExperience = 20
        self.csKnowledge = {
            "CSU1":u1Know,
            "CSU2":u2Know,
            "CSU3":u3Know,
            "CSU4":u4Know,
            "CSU5":u5Know,
            "CSU6":u6Know
            }
        self.otherKnowledge = {
            "English":engKnow,
            "Philosophy of Applied Mathematical Cognitive Biomechatronics and Foundations of Financial Aerospacial Agricultural Engineering":hidden
            }
        self.chatPhrase = "Today is a very great day with excellent weather "
    # textGenerate
    # @@para: testSent, sentLen
    # @@return: str
    # Description: a function that uses a neural network to generate new texts based on a given text
    def textGenerate(self, testSent, sentLen):
        for i in range(40):
            # first convert text to padded sequences
            testSeq = tok.texts_to_sequences([testSent])[0]
            testPad = pad_sequences([testSeq], maxlen=sentLen - 1, padding='pre')
            predictedNum = np.argmax(model.predict(testPad), axis=-1)
            predictedWord = ""
            # use the index, and go into the word index in tokenizer, to find the corresponding word
            for word, ind in tok.word_index.items():
                if ind == predictedNum:
                    predictedWord = word
                    break
            testSent += " " + predictedWord
        return testSent
    # teach
    # @@para: subject, sentLen
    # @@return: str
    # Description: a function that generates text based on a given topic of discussion
    def teach(self, subject, sentLen):
        # based on the subject one wants to learn, teach accordingly
        breaka = False
        while not breaka:
            if subject == "CSU1" or subject == "CSU2" or subject == "CSU3" or subject == "CSU4" or subject == "CSU5" or subject == "CSU6":
                toughtText = self.csKnowledge[subject]
                breaka = True
            # note that english and philosophy...engineering are both hidden options
            elif subject == "English":
                print("I am not J_II, but sure. I can teach you English.")
                toughtText = self.otherKnowledge[subject]
                breaka = True
            elif subject == "Philosophy of Applied Mathematical Cognitive Biomechatronics and Foundations of Financial Aerospacial Agricultural Engineering":
                toughtText = self.otherKnowledge[subject]
                breaka = True
            else:
                print("Invalid Answer.")
                subject = input("Please enter what you want me to teach. Enter 'CSU1', 'CSU2', 'CSU3', 'CSU4', 'CSU5', or 'CSU6'. \n")
        return self.textGenerate(toughtText, sentLen)
    # chat
    # @@para: userChat, sentLen
    # @@return: str
    # Description: a function that generates text based on what user chats about
    def chat(self, userChat, sentLen):
        testSent = self.chatPhrase + userChat
        return self.textGenerate(testSent, sentLen)
# create P_II AI teacher
p_II = AITeacher()
#-----------------------------------------------------------------------------------------------#
#################### Functions For Main Program ###############
#-----------------------------------------------------------------------------------------------#
# ask
# @@para: text, choice1, choice2
# @@return: int
# Description: a function for any types of asking for inputs
def ask(text, choice1, choice2):
    response = input(text)
    while True:
        if response == choice1:
            return 1
        elif response == choice2:
            return 2
        else:
            print("Invalid answer.")
            response = input(text)
# teacherOn
# @@para: sentLen
# @@return: void
# Description: the main user interface for the teacher
def teacherOn(sentLen):
    print("Welcome. I am your AI teacher, P_II.")
    while True:
        response = ask("Do you want to learn about something or just chat? Please enter 'learn' or 'chat'\n", 'learn','chat')
        if response == 1:
            subject = input("What do you want me to teach?\n")
            print(p_II.teach(subject, sentLen))
        else:
            chatStart = input("You can talk to me about anything.\n")
            print(p_II.chat(chatStart, sentLen))
        askQuit = ask("Do you want to continue talking to me? Enter 'yes' or 'no'.\n","yes", "no")
        if askQuit == 1:
            print("Great!")
        else:
            print("Ok. Have a BAD time. You missed a great opportunity by not talking to me, the great AI teacher. Bye bye.")
            reply = ask("Do you want to reply 'You are such a terrible teacher. P_I is better than you.' to P_II? Enter 'yes' or 'no'\n","yes", "no")
            if reply == 1:
                print("(P_II stared at you angrily, and left.)")
                return
            secret = input()
            if secret == "I want to learn machine learning (I want to make a better version of you).":
                print(p_II.textGenerate("To do machine learning, the most popular way today is to make a neural network.",sentLen))
            return

#-----------------------------------------------------------------------------------------------#
######################## Main Program ######################
#-----------------------------------------------------------------------------------------------#
print("\n\n\n\n\n############################")
teacherOn(maxLen)
