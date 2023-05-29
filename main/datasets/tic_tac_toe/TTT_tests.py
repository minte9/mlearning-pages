""" Tic tac toe (dataset test)

After encoding 'X'=2, 'O'=1 and ' '=0
For score we don't need to encode X=1 O=-1 draw=0
"""

import pathlib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import itertools as it

# Load dataset
DIR = pathlib.Path(__file__).resolve().parent
df = pd.read_csv(DIR / "data/ttt_dataset.csv")

# Encode lables
df_encoded = pd.DataFrame()
for col in df.columns:
    df_encoded[col] = LabelEncoder().fit_transform(df[col])

X = df_encoded.drop(columns=["score", "is_terminal"])
Y = pd.concat([df['score'], df_encoded['is_terminal']], axis=1)  # score 1,-1,0

# Fitting the model
dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(X, Y)

def get_legal_moves(board):
    moves = []
    for i,j in it.product(range(3), range(3)):
        if board[i, j] == " ":
            moves.append((i, j))
    return moves

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
        if B[i,j] == 'X': x_new.append(2)
        if B[i,j] == 'O': x_new.append(1)
        if B[i,j] == ' ': x_new.append(0)        
    return x_new

def play(board, player=True, expected=None):
    print("\nX" if player else "\nO", "move")

    best_move = None 
    best_score = float("-inf") if player else float("+inf") # initialize score

    for move in get_legal_moves(board):
        new_board = np.copy(board)
        new_board[move] = 'X' if player else 'O'

        x_new = convert_board(new_board)
        x_new = pd.DataFrame([x_new], columns=X.columns)
        score, is_terminal = dtree.predict(x_new)[0]

        if player == True and score > best_score:
                best_score = score
                best_move = move
            
        if player == False and score < best_score:
                best_score = score
                best_move = move

    board[best_move] = 'X' if player else 'O'
    show(board, best_move)

    if is_terminal == 1:
        if best_score ==  1: print('X won!')
        if best_score == -1: print('O won!')
        if best_score ==  0: print('Draw!')

        assert expected == best_score
        print('Test passed \n')
        return False

    play(board, not player, expected) # Recursive

games = [
    (
        np.array([
            ["X", "O", "X"], 
            ["O", "X", " "],
            ["O", "X", " "],]), False, 0 # Test 1
    ), (
        np.array([
            ["X", " ", " "],
            ["X", "O", " "],
            ["O", " ", " "],]), True, 0 # Test 2
    ), (
        np.array([
            [" ", "O", "X"],
            ["O", "X", " "],
            [" ", "X", " "],]), False, 0 # Test 3
    ), (
        np.array([
            ["X", "O", " "],
            ["X", "X", "O"],
            ["O", " ", " "],]), True, 1 # Test 4
    ), (
        np.array([
            ["X", "O", "X"],
            ["X", " ", " "],
            ["O", " ", "O"],]), True, 0 # Test 5
    ), (
        np.array([
            ["X", "O", " "],
            [" ", " ", " "],
            [" ", " ", " "],]), True, 1 # Test 6
    ), (
        np.array([
            ["X", " ", " "],
            [" ", " ", " "],
            [" ", " ", " "],]), False, 0 # Test 7
    ), (
        np.array([
            [" ", " ", " "],
            [" ", "X", " "],
            [" ", " ", " "],]), False, 0 # Test 8
    ), (
        np.array([
            ["O", " ", "X"],
            [" ", "X", " "],
            ["O", " ", " "],]), True, 0 # Test 9
    ),
    (
        np.array([
            [" ", " ", " "],
            [" ", " ", " "],
            ["O", " ", "X"],]), True, 1 # Test 10
    ),
]

i = 0
for board, player, expected in games:
    i = i + 1

    print("----------------------------- Test", i)
    print('X' if player else 'O', 'to move')
    show(board)
    play(board, player, expected)
