import math
import numpy as np

def serialize(board):
    temp = np.zeros((4, 4))
    for x in range(4):
        for y in range(4):
            if board[x][y] != 0:
                temp[x][y] = math.log2(board[x][y])
    return temp.reshape((-1))

ar = [[2, 4, 8, 16],
 [32, 64, 128, 256],
 [512, 1024, 2048, 4096,],
 [0, 0, 0, 0]]
print(ar)
print(serialize(ar))