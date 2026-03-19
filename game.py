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
        return self.board.copy()

    def step(self, action):
        """
        Rewrite gameplay
        # Agent move
        self.board[action] = 1

        if self.check_winner(1):
            return self.board.copy(), 1, True

        if len(available_actions()) == 0:
            return self.board.copy(), 0, True

        # Opponent (random)
        opp_action = np.random.choice(available_actions())
        self.board[opp_action] = -1

        if self.check_winner(-1):
            return self.board.copy(), -1, True

        if len(available_actions()) == 0:
            return self.board.copy(), 0, True
        """
        return self.board.copy(), 0, False

    def spawn(self):
        locations = []
        for x in range(4):
            for y in range(4):
                if self.board[x][y] == 0:
                    locations.append((x, y))
        l = random.choice(locations)
        self.board[l[0]][l[1]] = 2

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
game.spawn()
game.spawn()
print(game.check_lose())
print(game.board)