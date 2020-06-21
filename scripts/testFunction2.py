import sys
import numpy as np
from testClass import TestDemo
from testFunction import func

def infer(modelPath, imagePath):
	testDeme = TestDemo(modelPath)
	testDeme.evaluate(imagePath)

def calculate(imagePath):
	if (func(imagePath)):
		return 1
	else:
		return 0