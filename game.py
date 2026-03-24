import numpy as np
import random

class Game:
    def __init__(self):
        self.board = None
        self.done = None
        self.reset()

    def reset(self):
        self.board = np.zeros((4, 4))
        self.done = False
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
                    reward += self.board[y][x] * 0.00001
                    break
        return reward

    def monotonicity(self):
        score = 0
        for row in self.board:
            if all(row[i] <= row[i + 1] for i in range(3)) or \
                    all(row[i] >= row[i + 1] for i in range(3)):
                score += 1
        for col in self.board.T:
            if all(col[i] <= col[i + 1] for i in range(3)) or \
                    all(col[i] >= col[i + 1] for i in range(3)):
                score += 1
        return score

    def step(self, action):
        self.board = np.rot90(self.board, action)
        self.compress()
        merge_reward = self.merge()
        self.compress()
        self.board = np.rot90(self.board, -action)
        empty_tiles = np.sum(self.board == 0)
        max_tile = np.max(self.board)
        reward = merge_reward * 10 + max_tile * 0.1 + empty_tiles * 0.3 + self.monotonicity() * 0.5
        if self.check_lose() or not self.spawn():
            return self.board.copy(), reward - 10, True
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