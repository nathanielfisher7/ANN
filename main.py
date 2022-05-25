import functions.ANNop as ANNop
import numpy as np




testANN = ANNop.ANN(8, 0, 8)

testDataMatrix, testRowLabels, testColLabels, testNormalizedDataMatrix = testANN.preProcess("trainData.txt")

ANNop.printMatrix(testDataMatrix, testRowLabels, testColLabels)

# traindata = functions.datainit.readfile('traindata.txt')
