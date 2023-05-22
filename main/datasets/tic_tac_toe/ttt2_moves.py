""" Minimax algorithm
Color last move, dictionary refactoring
"""

import pathlib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# ------------------------------------------------------------

# Load dataset
DIR = pathlib.Path(__file__).resolve().parent
df = pd.read_csv(DIR / "data/tic-tac-toe-endgame.csv")

def add_draw_class(df):
    for i in range(len(df)):
        win = False

        for j in range(3):
            if df.iloc[i][j] == df.iloc[i][j+1] == df.iloc[i][j+2] and df.iloc[i][j+1] != 'b':
                win = True # horizontal win

        for j in range(3):
            if df.iloc[i][j] == df.iloc[i][j+3] == df.iloc[i][j+6] and df.iloc[i][j+1] != 'b':
                win = True # vertical win
        
        if df.iloc[i][0] == df.iloc[i][4] == df.iloc[i][9] or \
            df.iloc[i][2] == df.iloc[i][4] == df.iloc[i][6] and df.iloc[i][4] != 'b':
                win = True  # diagonal win

        if not win:
            df.loc[i]['V10'] = 'draw'
    return df

df = add_draw_class(df)

# Encode lables
LE = LabelEncoder()
df_encoded = pd.DataFrame()
for col in df.columns:
    df_encoded[col] = LE.fit_transform(df[col])

# Train data
X = df_encoded.iloc[:,0:-1]
Y = df_encoded['V10']

# Fitting the model
dtree_model = DecisionTreeClassifier(random_state=42)
dtree_model.fit(X, Y)

# ------------------------------------------------------------

def get_legal_moves(board):
    moves = []
    for i in range(3):
        for j in range(3):
            if board[i, j] == " ": 
                moves.append((i, j))
    return moves

def evaluate_score(board):
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] and board[i][0] != " ":
            return 1 if board[i][0] == 'X' else -1 # horizontal win score
    for i in range(3):
        if board[0][i] == board[1][i] == board[2][i] and board[0][i] != " ":
            return 1 if board[0][i] == 'X' else -1 # vertical win score

    if (board[0][0] == board[1][1] == board[2][2] or \
        board[0][2] == board[1][1] == board[2][0]) and board[1][1] != " ":
            return 1 if board[1][1] == "X" else -1 # diagonal win score 
    return 0

def is_terminal_state(board):
    if (evaluate_score(board)) ==  1:    return True # X win
    if (evaluate_score(board)) == -1:    return True # O win
    if len(get_legal_moves(board)) == 0: return True # Draw
    return False

def show(board, move=None):
    board_ = np.copy(board).tolist() 
    if move: 
        i, j = move
        CYAN, ENDC = '\033[96m', '\033[0m'
        board_[i][j] = CYAN + board[move] + ENDC # colored move
        
    for i in range(3):
        print(" ", board_[i][0], "|", board_[i][1], "|", board_[i][2])
        print(" ---+---+---") if i < 2 else ""

# Convert board to match model state format like 
def convert_board(new_board): # [2,1,0,2,2,1,1,0,0]
    new_state = []
    for row in new_board:
        for move in row:
            if move == ' ': v = 0
            if move == 'O': v = 1
            if move == 'X': v = 2
            new_state.append(v)
    return new_state

def convert_dtree_score(score):
    if score == 0: return  0 # Draw
    if score == 1: return -1 # O win
    if score == 2: return 1 # X win

# ------------------------------------------------------------

def play(board, player=True, expected=None):
    print("\nX" if player else "\nO", "move")

    best_move = None 
    best_score = float("-inf") if player else float("+inf") # initialize score

    for move in get_legal_moves(board): # possible moves
        new_board = np.copy(board)
        new_board[move] = 'X' if player else 'O'

        if is_terminal_state(new_board): 
            best_move = move
            best_score = evaluate_score(new_board)
            break

        # Predict - children scores
        x_new = convert_board(new_board)
        x_new = pd.DataFrame([x_new], columns=X.columns)
        y_pred = dtree_model.predict(x_new)[0]  # 0,  1, 2
        score_ = convert_dtree_score(y_pred)    # 0, -1, 1

        if player == True:
            if score_ > best_score:
                best_score = score_ # child best score
                best_move = move    # parent move
            
        if player == False:
            if score_ < best_score:
                best_score = score_
                best_move = move

    board[best_move] = 'X' if player else 'O'
    show(board, best_move)

    if is_terminal_state(board):
        if best_score ==  1: print('X won!')
        if best_score == -1: print('O won!')
        if best_score ==  0: print('Draw!')

        assert expected == best_score
        print('Test passed \n')
        return False # Base case

    play(board, not player, expected) # Recursive case

# ------------------------------------------------------------

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
    # if i != 6: continue

    print("----------------------------- Test", i)
    print('X' if player else 'O', 'to move')
    show(board)
    play(board, player, expected)