import numpy as np




def printMatrix(data_matrix, rlabel_matrix, clabel_matrix):
    # Get the size of all three matrices
    tsize = data_matrix.shape
    rsize = rlabel_matrix.shape
    csize = clabel_matrix.shape
    
    #Check for errors
    if (tsize[0] != rsize[0]):
        print("{}".format("ERROR: Row sizes of row label matrix and the data matrix don't match\n", end=''))
        return
    elif(tsize[1] != csize[1]):
        print("{}".format("ERROR: Column sizes of column label matrix and the data matrix don't match\n", end=''))
        return
    else:
        # If no errors were found, print out row size and column size
        print("Number of Rows = %d\nNumber of Columns = %d" % (tsize[0]+1, tsize[1]+1))
    # print(clabel_matrix[0, 0]
    # print(rlabel_matrix.shape)
    # print(tsize)
    print("{}".format("Training Data\n-------------------------------------------------------------------"))
    '''
        Properly output a formatted training data table
    '''
    for x in range(tsize[0]):
        if (x == 0):
            print('{:<4}'.format(''), end='')
            for z in range(tsize[1] - 1):
                print('{:<14}'.format(clabel_matrix[0, z]), end='')
        else:
            print('{:<4}'.format(rlabel_matrix[x - 1, 0]), end='')
        for y in range(tsize[1] - 1):
            if (x == 0):
                pass
            else:
                print('{:<14}'.format(data_matrix[x - 1, y]), end='')
        print('')
    return


def readfile(fileName):
    # Generate matrix of the data contained within the file selected.
    file_matrix = np.asmatrix(np.genfromtxt(fileName, dtype=str))
    
    # Find the row and column amount
    tsize = file_matrix.shape
    
    # Separate the data, and row and column labels into separate matrices
    data_matrix = (file_matrix[1:tsize[0], 1:tsize[1]]).astype(np.float)
    rlabel_matrix = file_matrix[1:tsize[0], 0]
    clabel_matrix = file_matrix[0, 1:tsize[1]]

    # Print out the matrix
    printMatrix(data_matrix, rlabel_matrix, clabel_matrix)

    return data_matrix, rlabel_matrix, clabel_matrix


def normalize(data_matrix):
    # Gather the row and column size of the data matrix
    [row_amount, column_amount] = data_matrix.shape

    # Square the values in the data matrix
    data_matrix_square = np.square(data_matrix)

    # Produce a sum matrix of the squared data matrix
    moments_sum = sum(data_matrix_square)

    # Produce a matrix of the RMS values
    moments_RMS = np.sqrt((moments_sum / row_amount))

    # Produce a normalized data matrix
    normalized_data_matrix = data_matrix / moments_RMS

    # Return the normalized data matrix
    return normalized_data_matrix

def train_Pattern(normalized_data_matrix,class_labels):


    pass

class ANN():

    def __init__(self, inputNodeNum, hiddenNodeNum, outputNodeNum):
        self.inputNodeNum = inputNodeNum
        self.hiddenNodeNum = hiddenNodeNum
        self.outputNodeNum = outputNodeNum

    def preProcess(self, trainDataFile):
        trainDataMatrix, rowLabels, colLabels, = readfile(trainDataFile)
        normalizedTrainDataMatrix = normalize(trainDataMatrix)
        return trainDataMatrix, rowLabels, colLabels, normalizedTrainDataMatrix
        # train_Pattern(normalizedTrainDataMatrix, rowLabels)
