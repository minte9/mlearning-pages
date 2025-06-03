## Tic Tac Toe / Decision Tree

### Raw dataset

Instances: 958 (legal tic-tac-toe endgame boards)
Attributes: 9, each corresponding to one tic-tac-toe square

### Full dataset

For each cell on the board, there are three possible states: empty, X, or O. 
Since there are nine cells in total, the total number of unique game states 
can be calculated as: 
    3^9 = 19,683

This calculation represents all possible combinations with repetitions 
of the 3 states for each of the 9 cells. 

However, not all of these states are valid or reachable during a game of Tic-Tac-Toe.
Only 5478 are possible.

### References

[TTT with decision tree](https://www.kaggle.com/code/pulkitmundra/tictactoe-with-decision-tree/notebook) notebook  
[TTT dataset 1](https://www.kaggle.com/datasets/aungpyaeap/tictactoe-endgame-dataset-uci) dataset  
[TTT dataset 2](https://www.kaggle.com/datasets/somesh24/tictactoe)
[Computer LEARNS Tic-Tac-Toe](https://amit9oct.github.io/2020-08-12-LearnTicTacToe/)