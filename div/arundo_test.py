import os
import numpy as np
import pandas as pd
from keras import models
import json
import pickle

os.chdir('..')
#file = json.load('data2.json')
#print(file)

time = "13:61"


def validTime(time):
    for index, integer in enumerate(time):
        print(index)
        print(integer)
        if index == 0:
            if float(integer) > 2:
                return False
            elif float(integer) == 2 and int(time[1]) > 3:
                return False
        elif index == 2:
            if float(integer) > 5:
                return False


def runLengthEncoding(inputString):
    string = ''
    sample_char = ''
    counter = 1
    for char in inputString:
        print(char)
        if char == sample_char:
            counter += 1
        else:
            string = string + str(counter) + char
            counter = 1
        sample_char = char
    return string


runLengthEncoding('aaaaabbbbc')