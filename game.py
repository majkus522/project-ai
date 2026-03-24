import numpy as np
import random

class Game:
    def __init__(self):
        self.board = None
        self.done = None
        self.max = 0
        self.reset()

    def reset(self):
        self.board = np.zeros((4, 4))
        self.done = False
        self.max = 0
        self.spawn()
        self.spawn()
        return self.board.copy()

    def compress(self):
        for y in range(2, -1, -1):
            for x in range(4):
                if self.board[y][x] == 0:
                    continue
                for dy in range(3, y, -1):
                    if self.board[dy][x] == 0:
                        self.board[dy][x] = self.board[y][x]
                        self.board[y][x] = 0
                        break

    def merge(self):
        reward = 0
        for y in range(3, 0, -1):
            for x in range(4):
                if self.board[y][x] == 0:
                    continue
                if self.board[y][x] == self.board[y - 1][x]:
                    self.board[y][x] += 1
                    self.board[y - 1][x] = 0
                    if self.board[y][x] > self.max:
                        self.max = self.board[y][x]
                    reward += self.board[y][x] * 0.1
                    break
        return reward

    def step(self, action):
        self.board = np.rot90(self.board, action)
        self.compress()
        reward = self.merge()
        self.compress()
        self.board = np.rot90(self.board, -action)
        if self.check_lose() or not self.spawn():
            return self.board.copy(), reward + self.max, True
        return self.board.copy(), reward, False

    def spawn(self):
        locations = []
        for x in range(4):
            for y in range(4):
                if self.board[x][y] == 0:
                    locations.append((x, y))
        if len(locations) == 0:
            return False
        l = random.choice(locations)
        self.board[l[0]][l[1]] = 1
        return True

    def check_lose(self):
        for x in range(4):
            for y in range(4):
                if self.board[x][y] == 0:
                    continue
                if x - 1 >= 0:
                    if self.board[x][y] == self.board[x - 1][y] or self.board[x - 1][y] == 0:
                        return False
                if y - 1 >= 0:
                    if self.board[x][y] == self.board[x][y - 1] or self.board[x][y - 1] == 0:
                        return False
                if x + 1 <= 3:
                    if self.board[x][y] == self.board[x + 1][y] or self.board[x + 1][y] == 0:
                        return False
                if y + 1 <= 3:
                    if self.board[x][y] == self.board[x][y + 1] or self.board[x][y + 1] == 0:
                        return False
        return True

game = Game()
while True:
    print(game.board)
    a = int(input("Input: "))
    ret = game.step(a)
    if ret[2]:
        break