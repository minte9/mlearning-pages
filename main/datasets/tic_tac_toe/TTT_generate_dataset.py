""" Tic tac toe (generate dataset)

Generate the complete tree using minimax algorithm.
Even though applying machine learning to a problem which can be easily solved 
by deterministic methods does not add much value, it can still be considered 
a good problem for learning ML.
"""

import csv
import numpy as np
import copy
import pathlib

dataset = []
board = np.array([
    [" ", " ", " "],
    [" ", " ", " "],
    [" ", " ", " "]])

def get_legal_moves(board):
    moves = []
    for i in range(3):
        for j in range(3):
            if board[i, j] == " ":
                moves.append((i, j))
    return moves

def evaluate_score(board):

    # horizontal win score
    for i in range(3): 
        if board[i][0] == board[i][1] == board[i][2] and board[i][0] != " ":
            return 1 if board[i][0] == 'X' else -1

    # vertical win score
    for i in range(3): 
        if board[0][i] == board[1][i] == board[2][i] and board[0][i] != " ":
            return 1 if board[0][i] == 'X' else -1

    # diagonal win score
    if (board[0][0] == board[1][1] == board[2][2] or \
        board[0][2] == board[1][1] == board[2][0]) and board[1][1] != " ":
            return 1 if board[1][1] == "X" else -1

    return 0

def is_terminal_state(board):
    if (evaluate_score(board)) ==  1: return True # X win
    if (evaluate_score(board)) == -1: return True # O win
    if len(get_legal_moves(board)) == 0: return True 
    return False

def minimax(board, player=True, alpha=float('-inf'), beta=float('inf')):
    
    best_move = None 
    best_score = float("-inf") if player else float("+inf") # initialize score

    for move in get_legal_moves(board): # possible moves

        new_board = np.copy(board)
        new_board[move] = 'X' if player else 'O'
        
        if is_terminal_state(new_board): 
            dataset.append((new_board.flatten(), evaluate_score(new_board), True, move))
            return move, evaluate_score(new_board) # Base case

        # Recursive case
        move_, score_ = minimax(new_board, not player, alpha, beta)
        dataset.append((new_board.flatten(), score_, False, move))

        if player == True:
            if score_ > best_score:
                best_score = score_ # child best score
                best_move = move    # parent move
            alpha = max(alpha, score_)
            
        if player == False:
            if score_ < best_score:
                best_score = score_
                best_move = move
            beta = min(beta, score_)
        
        if alpha >= beta: # pruning condition
            break

    return best_move, best_score

def generate_dataset(board, player=True):

    for move in get_legal_moves(board):
        new_board = copy.deepcopy(board)
        new_board[move] = 'X' if player else 'O'
        
        _,_ = minimax(new_board, not player)
 
generate_dataset(board)

DIR = pathlib.Path(__file__).resolve().parent
FILE = DIR / "data/ttt_dataset.csv"
with open(FILE, 'w') as file:

    writer = csv.writer(file)
    writer.writerow(['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'score', 'is_terminal'])

    for value in dataset:
        line = (*value[0], value[1], value[2]) # unpacks the elements of row into separate values

        writer.writerow(line) 
        print(*line)
