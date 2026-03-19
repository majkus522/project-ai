import numpy as np

class Game:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros(9)  # 0 empty, 1 agent, -1 opponent
        self.done = False
        return self.board.copy()

    def available_actions(self):
        return [i for i in range(9) if self.board[i] == 0]

    def step(self, action):
        # Agent move
        self.board[action] = 1

        if self.check_winner(1):
            return self.board.copy(), 1, True

        if len(self.available_actions()) == 0:
            return self.board.copy(), 0, True

        # Opponent (random)
        opp_action = np.random.choice(self.available_actions())
        self.board[opp_action] = -1

        if self.check_winner(-1):
            return self.board.copy(), -1, True

        if len(self.available_actions()) == 0:
            return self.board.copy(), 0, True

        return self.board.copy(), 0, False

    def check_winner(self, player):
        b = self.board.reshape(3, 3)
        for i in range(3):
            if all(b[i, :] == player) or all(b[:, i] == player):
                return True
        if b[0,0] == b[1,1] == b[2,2] == player:
            return True
        if b[0,2] == b[1,1] == b[2,0] == player:
            return True
        return False