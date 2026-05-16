import random, json, noughts_and_crosses

def winner(board):
    wins = [
        (0,1,2), (3,4,5), (6,7,8),
        (0,3,6), (1,4,7), (2,5,8),
        (0,4,8), (2,4,6)
    ]

    for a, b, c in wins:
        if board[a] != 0 and board[a] == board[b] == board[c]:
            return board[a]

    return 0


def generate_reachable_states():
    states = set()

    def recurse(board, player):
        states.add(tuple(board))

        if winner(board) != 0:
            return

        if 0 not in board:
            return

        for i in range(9):
            if board[i] == 0:
                new_board = board[:]
                new_board[i] = player
                recurse(new_board, -player)

    recurse([0] * 9, 1)

    return [list(state) for state in states]

#not op
with open("ai/NoughtsAndCrosses/data2.json", "r") as data:
    def n(combination):
        '''-1 -> 1, 1 -> -1'''
        combination = combination.split(',')
        combination = [str(int(k) * -1) for k in combination]
        return ''.join(k + ',' for k in combination)[:-1]
    keys = []
    values = []
    data = json.load(data)
    for i in data.keys():
        keys.append(n(i))
        values.append(data[i])
    out = dict(zip(keys, values))
# with open("ai/NoughtsAndCrosses/data2.json", 'w') as data:
#     json.dump(out, data, indent=4)

boards = generate_reachable_states()

to_write = {
    ",".join(map(str, board)): [
        i for i in range(9) if board[i] == 0 for _ in range(3)
    ]
    for board in boards
}

# with open("ai/NoughtsAndCrosses/data.json", "a") as data:
#     json.dump(to_write, data, indent=4)

def get_computer_move(combination) -> int:
    try:
        with open("ai/NoughtsAndCrosses/data.json", 'r') as data:
            data = json.load(data)
        combination = ''.join([str(''.join([str(j) + ',' for j in k])) for k in combination])[:-1]
        moves = data[combination]
        return random.choice(moves)
    except IndexError:
        return -1
    except:
        return -2


def get_computer2_move(combination) -> int:
    try:
        with open("ai/NoughtsAndCrosses/data2.json", 'r') as data:
            data = json.load(data)
        combination = ''.join([str(''.join([str(j) + ',' for j in k])) for k in combination])[:-1]
        moves = data[combination]
        return random.choice(moves)
    except IndexError:
        return -1
    except:
        return -2

def help1(a: list[list[int]]) -> str:
    '''converts board to string format in json'''
    return ''.join([str(''.join([str(j) + ',' for j in k])) for k in a])[:-1]
# mprint(get_computer_move(
#     [
#         [0, 1, 1],
#         [0, -1, 0],
#         [0, 0, 0]
#     ]
# ))

convert = {
    0: [0, 0],
    1: [0, 1],
    2: [0, 2],
    3: [1, 0],
    4: [1, 1],
    5: [1, 2],
    6: [2, 0],
    7: [2, 1],
    8: [2, 2],
}
reverse_convert = [
    [0, 0],
    [0, 1],
    [0, 2],
    [1, 0],
    [1, 1],
    [1, 2],
    [2, 0],
    [2, 1],
    [2, 2]
]
global p #########################################################################################################
p = False
def mprint(*args, **kwargs):
    global p
    if p == True:
        print(*args, **kwargs)
errors = 0
for y in range(0, 2):
    game = noughts_and_crosses.Game()
    a = True
    computer_moves = []
    computer_won = 'draw'
    computer2_moves = []
    count = 1
    last_move = 0
    while a == True or count < 9:
        
        if game.turn == 1:
            mprint(*[k for k in game.board], game.turn, sep='\n')
            mprint('Computer\'s turn')
            c = get_computer_move(game.board)
            if c == -1:
                computer_won = 'draw'
                break
            if c == -2:
                errors += 1
            d = help1(game.board)
            b = game.play(convert[int(c)])
            computer_moves.append((c, d))
            last_move = c
            if b == 1:
                mprint('Computer won')
                computer_won = 'win'
                a = False
                break

        if game.turn == -1:
            # mprint(*[k for k in game.board], game.turn, sep='\n')
            # mprint('Your turn')
            # last_move = input("enter the coords of the square which you want to place: ")
            # if game.play(eval(last_move)) == -1:
            #     mprint("you win")
            #     computer_won = 'lose'
            #     a = False
            #     break
            mprint(*[k for k in game.board], game.turn, sep='\n')
            mprint('Other computer\'s turn')
            c = get_computer2_move(game.board)
            d = help1(game.board)
            if c == -1:
                computer_won = 'draw'
                break
            if c == -2:
                errors += 1
            b = game.play(convert[int(c)])
            computer2_moves.append((c, d))
            last_move = c
            if b == -1:
                mprint('Other computer won')
                computer_won = 'false'
                a = False
                break

        count += 1

    if computer_won == '':
        computer_won = 'draw'

    mprint(computer_moves)
    #TRAIN:
    #train 2nd computer
    with open("ai/NoughtsAndCrosses/data2.json", 'r') as data:
        data = json.load(data)
        for i in range(0, len(computer2_moves)):
            if computer_won == 'win':
                for j in range(0, i * 2):
                    data[computer2_moves[i][1]].append(computer2_moves[i][0])
            if computer_won == 'draw':
                data[computer2_moves[i][1]].append(computer2_moves[i][0])
            if computer_won == 'lose':
                if i == len(computer2_moves) - 1:
                    data[computer2_moves[i][1]] = [reverse_convert.index(eval(last_move))]
                else:
                    for i in range(0, i * 2):
                        try:
                            data[computer2_moves[i][i]].remove(computer2_moves[i][0])
                        except:
                            pass
    with open("ai/NoughtsAndCrosses/data2.json", 'w') as x:
        json.dump(data, x, indent=4)


    #train 1st computer
    with open("ai/NoughtsAndCrosses/data.json", 'r') as data:
        data = json.load(data)
        for i in range(0, len(computer_moves)):
            if computer_won == 'win':
                for j in range(0, i * 2):
                    data[computer_moves[i][1]].append(computer_moves[i][0])
            if computer_won == 'draw':
                data[computer_moves[i][1]].append(computer_moves[i][0])
            if computer_won == 'lose':
                if i == len(computer_moves) - 1:
                    data[computer_moves[i][1]] = [reverse_convert.index(eval(last_move))]
                else:
                    for i in range(0, i * 2):
                        try:
                            data[computer_moves[i][i]].remove(computer_moves[i][0])
                        except:
                            pass
    with open("ai/NoughtsAndCrosses/data.json", 'w') as x:
        json.dump(data, x, indent=4)
        mprint(data == x, '/', computer_won)
    mprint("trained", y)
    print()
mprint('Errors: ', errors)


