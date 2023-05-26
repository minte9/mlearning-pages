## Tic Tac Toe / Decision Tree

### Raw dataset

Instances: 958 (legal tic-tac-toe endgame boards)
Attributes: 9, each corresponding to one tic-tac-toe square

### Full dataset

To determine the number of reachable game states, 
we can analyze the possibilities at each step of the game.

In the first move, there are 9 possible positions for the first player (X).
After the first move, the second player (O) has 8 possible positions remaining.
For each subsequent move, the number of available positions decreases by 1.
Therefore, the total number of reachable game states can be calculated as follows:

9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1 = 9! = 362,880

### References

[TTT with decision tree](https://www.kaggle.com/code/pulkitmundra/tictactoe-with-decision-tree/notebook) notebook  
[TTT dataset 1](https://www.kaggle.com/datasets/aungpyaeap/tictactoe-endgame-dataset-uci) dataset  
[TTT dataset 2](https://www.kaggle.com/datasets/somesh24/tictactoe)
[TTT dataset 3](https://amit9oct.github.io/2020-08-12-LearnTicTacToe/)