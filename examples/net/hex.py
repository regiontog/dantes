import numpy as np


def cells(n):
    for i in range(0, n):
        for j in range(0, i+1):
            yield i - j, j

    for j in range(1, n):
        for i in reversed(range(j, n)):
            yield i, n - i + j - 1


NEIGHBORS = [
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (-1, -1),
    (1, 1),
]


def neighbors(n, cell):
    return filter(lambda neighbor: 0 <= neighbor[0] < n and 0 <= neighbor[1] < n,
                  map(lambda offset: (
                      cell[0] + offset[0], cell[1] + offset[1]), NEIGHBORS)
                  )


class Hex:
    TOPLEFT = 3
    TOPRIGHT = 4
    BOTTOMLEFT = 5
    BOTTOMRIGHT = 6

    def __init__(self, n):
        self.n = n
        self.all_actions = list((i, j) for i in range(0, n)
                                for j in range(0, n))

        self.action_indices = {action: i for i,
                               action in enumerate(self.all_actions)}

        self.cells = list(cells(n))

        self.neighbors = {cell: list(neighbors(n, cell))
                          for cell in self.all_actions}

    def initial_state(self):
        return (1, np.zeros((self.n, self.n)), None)

    def action_from_distribution(self, state, distribution):
        actions = list(self.action_space(state))

        distribution = np.take(distribution, list(map(
            lambda action: self.action_indices[action], actions)))

        return actions[np.argmax(distribution)]

    def take_action(self, state, action):
        (player, board, _) = state
        board = np.copy(board)

        board[action[0], action[1]] = player
        return (2 if player == 1 else 1, board, action)

    def action_space(self, state):
        return filter(lambda action: state[1][action[0],
                                              action[1]] == 0, self.all_actions)

    def walls(self, action):
        (r, c) = action

        if r == 0:
            yield Hex.TOPRIGHT
        if r == self.n - 1:
            yield Hex.BOTTOMLEFT
        if c == 0:
            yield Hex.TOPLEFT
        if c == self.n - 1:
            yield Hex.BOTTOMRIGHT

    def result(self, state):
        (player, board, prev_action) = state

        if prev_action is None:
            return 0

        maybe_winner = 2 if player == 1 else 1

        connected = [prev_action]
        visited = set()

        left_wall = False
        right_wall = False

        while len(connected) > 0:
            action = connected.pop()
            visited.add(action)

            if board[action[0], action[1]] == maybe_winner:
                for wall in self.walls(action):
                    if wall == Hex.TOPLEFT and maybe_winner == 2:
                        left_wall = True
                    elif wall == Hex.BOTTOMRIGHT and maybe_winner == 2:
                        right_wall = True
                    elif wall == Hex.TOPRIGHT and maybe_winner == 1:
                        right_wall = True
                    if wall == Hex.BOTTOMLEFT and maybe_winner == 1:
                        left_wall = True

                if left_wall and right_wall:
                    return maybe_winner

                connected.extend(
                    filter(lambda action: action not in visited, self.neighbors[action]))

        return 0

    def print(self, game):
        cell_index = 0

        def color(cell):
            if cell == 1:
                return 'r'
            elif cell == 2:
                return 'b'
            else:
                return ' '

        def next_cell():
            nonlocal cell_index
            cell = self.cells[cell_index]
            cell_index += 1
            return game[1][cell[0], cell[1]]

        print(" " * (self.n * 2 + 1), "_", sep="")

        for i in range(1, self.n):
            print(" " * (self.n * 2 + 1 - 2 * i), "_", "_".join(map(lambda _: "/" +
                                                                    color(next_cell()) + "\\", range(0, i))), "_", sep="")

        print(" " * 2, "_".join(map(lambda _: "/" +
                                    color(next_cell()) + "\\", range(0, self.n))), sep="")

        def bottom(i):
            yield "\\_/"
            for j in range(0, i - 1):
                yield color(next_cell())
                yield "\\_/"

        for i in reversed(range(1, self.n + 1)):
            print(" " * (self.n * 2 + 2 - 2 * i),
                  "".join(bottom(i)), sep="")
