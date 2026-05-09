class Game:
    def __init__(self):
        self.board = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]
        self.turn = 1
        # 1 = X, -1 = O

    def play(self, move: tuple[int] | list[int]):
        self.board[move[0]][move[1]] = self.turn
        if self.turn == 1:
            self.turn = -1
        else:
            self.turn = 1
        return self.check_win()

    def check_win(self):
        a = self.board
        #check horizontals
        if a[0][0] == a[0][1] == a[0][2] and a[0][0] != 0: #top row
            return a[0][0]
        elif a[1][0] == a[1][1] == a[1][2] and a[1][0] != 0: #middle row
            return a[1][0]
        elif a[2][0] == a[2][1] == a[2][2] and a[2][0] != 0: #bottom row
            return a[2][0]
        #check columns
        if a[0][0] == a[1][0] == a[2][0] and a[0][0] != 0: #left column
            return a[0][0]
        elif a[0][1] == a[1][1] == a[2][1] and a[0][1] != 0: #mid column
            return a[0][1]
        elif a[0][2] == a[1][2] == a[2][2] and a[0][2] != 0: #right column
            return a[0][2]
        #check diagonals
        if a[0][0] == a[1][1] == a[2][2] and a[0][0] != 0:
            return a[0][0]
        elif a[0][2] == a[1][1] == a[2][0] and a[1][1] != 0:
            return a[1][1]
        
        #no winner:
        return 0


a = Game()

