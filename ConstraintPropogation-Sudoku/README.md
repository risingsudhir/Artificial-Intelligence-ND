# Artificial Intelligence Nanodegree
## Introductory Project: Diagonal Sudoku Solver

# Question 1 (Naked Twins)
Q: How do we use constraint propagation to solve the naked twins problem?  
A: The game applies the constraints on possible digits of sudoku boxes that no two boxes can have same value if they are in the same unit. For a diagonal sudoku, a unit is either one complete 
row, or a complete column, or a 3x3 square matrix, or diagonals of the complete square. At any point of time, if two boxes of the game appear to have exactly same digits as their possiblity, 
then under the digits constraint, only these two boxes can have these two digits. Although it does not reduce the puzzle for twin boxes, it gives the hint that 
none of the boxes in the same unit can have these two digits, except twin boxes. By applying this constraint, we remove these two digits from other boxes in the same unit as elimination strategy.
This is how we use constraint propogation to solve the naked twins problem.

# Question 2 (Diagonal Sudoku)
Q: How do we use constraint propagation to solve the diagonal sudoku problem?  
A: The game applies the constraints on possible digits of sudoku boxes that no two boxes can have same value if they are in the same unit. For a non-diagonal sudoku, a unit is either one complete 
row, or a complete column, or a 3x3 square matrix. To solve the diagonal sudoku, we augument the constraints by adding diagonals of the complete square as unit. This adds the constraint that across diagonals, 
each boxe should have unique number from 1-9. With augumented constraint, the problem reduces to standard sodoku problem which can be solved by constraint propogation and search strategy and
elimination strategies (eliminate, naked twins, only choice) propogate this diagonal constraint to reduce the puzzle.


### Install

This project requires **Python 3**.

We recommend students install [Anaconda](https://www.continuum.io/downloads), a pre-packaged Python distribution that contains all of the necessary libraries and software for this project. 
Please try using the environment we provided in the Anaconda lesson of the Nanodegree.

##### Optional: Pygame

Optionally, you can also install pygame if you want to see your visualization. If you've followed our instructions for setting up our conda environment, you should be all set.

If not, please see how to download pygame [here](http://www.pygame.org/download.shtml).

### Code

* `solution.py` - You'll fill this in as part of your solution.
* `solution_test.py` - Do not modify this. You can test your solution by running `python solution_test.py`.
* `PySudoku.py` - Do not modify this. This is code for visualizing your solution.
* `visualize.py` - Do not modify this. This is code for visualizing your solution.

### Visualizing

To visualize your solution, please only assign values to the values_dict using the ```assign_values``` function provided in solution.py

### Submission
Before submitting your solution to a reviewer, you are required to submit your project to Udacity's Project Assistant, which will provide some initial feedback.  

The setup is simple.  If you have not installed the client tool already, then you may do so with the command `pip install udacity-pa`.  

To submit your code to the project assistant, run `udacity submit` from within the top-level directory of this project.  You will be prompted for a username and password.  If you login using google or facebook, visit [this link](https://project-assistant.udacity.com/auth_tokens/jwt_login for alternate login instructions.

This process will create a zipfile in your top-level directory named sudoku-<id>.zip.  This is the file that you should submit to the Udacity reviews system.

