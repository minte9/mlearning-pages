import pathlib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import itertools as it

# Load dataset
DIR = pathlib.Path(__file__).resolve().parent
df = pd.read_csv(DIR / "data/tic-tac-toe-amit9oc3.csv")

# Train data
X1 = df[1 == df.IsO].drop(columns=["IsO", "score"])  # O last move
X2 = df[0 == df.IsO].drop(columns=["IsO", "score"])  # X last move
Y1 = df[1 == df.IsO]['score']
Y2 = df[0 == df.IsO]['score']

# Fitting the models (X and O)
dtreeX = DecisionTreeClassifier()
dtreeO = DecisionTreeClassifier()
dtreeX.fit(X1, Y1)
dtreeO.fit(X2, Y2)

def get_legal_moves(board):
    moves = []
    for i,j in it.product(range(3), range(3)):
        if board[i, j] == " ":
            moves.append((i, j))
    return moves

def evaluate_score(B): # board

    for i in range(3):
        if B[i,0] == B[i,1] == B[i,2] and B[i,0] != " ":
            return 1 if B[i,0] == 'X' else -1  # horizontal win score

    for i in range(3):
        if B[0,i] == B[1,i] == B[2,i] and B[0,i] != " ":
            return 1 if B[0,i] == 'X' else -1  # vertical win score

    if (B[0,0] == B[1,1] == B[2,2] or
            B[0,2] == B[1,1] == B[2,0]) and B[1,1] != " ":
        return 1 if B[1,1] == "X" else -1  # diagonal win score 
    return 0

def is_terminal_state(board):
    if evaluate_score(board) == 1: return True  # X win
    if evaluate_score(board) == -1: return True  # O win
    if len(get_legal_moves(board)) == 0: return True  # Draw
    return False

def show(board, move=None):
    board_ = np.copy(board).tolist()
    if move:
        i, j = move
        CYAN, ENDC = '\033[96m', '\033[0m'
        board_[i][j] = CYAN + board[move] + ENDC  # colored move

    for i in range(3):
        print(" ", board_[i][0], "|", board_[i][1], "|", board_[i][2])
        if i < 2:
            print(" ---+---+---")

def convert_board(B):
    x_new = []
    
    for i,j in it.product(range(3), range(3)):
        if B[i,j] == 'X': x_new.append(1)
        if B[i,j] == 'O': x_new.append(0)
        if B[i,j] == ' ': x_new.append(0)

    for i,j in it.product(range(3), range(3)):
        if B[i,j] == 'O': x_new.append(1)
        if B[i,j] == 'X': x_new.append(0)
        if B[i,j] == ' ': x_new.append(0)
        
    return x_new

def play(board, player=True, expected=None):
    print("\nX" if player else "\nO", "move")

    best_move = None 
    best_score = float("-inf") if player else float("+inf") # initialize score
    
    players = [player, not player]
    moves = get_legal_moves(board)

    for player_, move in it.product(players, moves): # cartesian product
        new_board = np.copy(board)
        new_board[move] = 'X' if player_ else 'O'

        if is_terminal_state(new_board): 
            best_move = move
            best_score = evaluate_score(new_board)
            break

        dtree_model = dtreeX if player_ else dtreeO

        x_new = convert_board(new_board)
        x_new = pd.DataFrame([x_new], columns=X1.columns)
        score_ = dtree_model.predict(x_new)[0]

        if player_ == True:
            if score_ > best_score:
                best_score = score_
                best_move = move
            
        if player_ == False:
            if score_ < best_score:
                best_score = score_
                best_move = move

    board[best_move] = 'X' if player else 'O'
    show(board, best_move)

    if is_terminal_state(board):
        if best_score == 1:  print('X won!')
        if best_score == -1: print('O won!')
        if best_score == 0:  print('Draw!')

        assert expected == best_score
        print('Test passed \n')
        return False

    play(board, not player, expected)

games = [
    (
        np.array([
            ["X", "O", "X"], 
            ["O", "X", " "],
            ["O", "X", " "],]), False, 0
    ), (
        np.array([
            ["X", " ", " "],
            ["X", "O", " "],
            ["O", " ", " "],]), True, 0
    ), (
        np.array([
            [" ", "O", "X"],
            ["O", "X", " "],
            [" ", "X", " "],]), False, 0
    ), (
        np.array([
            ["X", "O", " "],
            ["X", "X", "O"],
            ["O", " ", " "],]), True, 1
    ), (
        np.array([
            ["X", "O", "X"],
            ["X", " ", " "],
            ["O", " ", "O"],]), True, 0
    ), (
        np.array([
            ["X", "O", " "],
            [" ", " ", " "],
            [" ", " ", " "],]), True, 1
    ), (
        np.array([
            ["X", " ", " "],
            [" ", " ", " "],
            [" ", " ", " "],]), False, 0
    ), (
        np.array([
            [" ", " ", " "],
            [" ", "X", " "],
            [" ", " ", " "],]), False, 0
    ), (
        np.array([
            ["O", " ", "X"],
            [" ", "X", " "],
            ["O", " ", " "],]), True, 0
    ),
    (
        np.array([
            [" ", " ", " "],
            [" ", " ", " "],
            ["O", " ", "X"],]), True, 1
    ),
]

i = 0
for board, player, expected in games:
    i = i + 1

    print("----------------------------- Test", i)
    print('X' if player else 'O', 'to move')
    show(board)
    play(board, player, expected)
