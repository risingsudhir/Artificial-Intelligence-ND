"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random
#import time

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Heuristic based on player's move imbalance and keeping optimal Manhattan distance from opponent
        
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
    
    # get the Manhattan distance between players
    p_y, p_x = game.get_player_location(player)
    o_y, o_x = game.get_player_location(game.get_opponent(player))
    distance = abs(p_y - o_y) + abs(p_x - o_x)
    
    # find move imbalance
    my_moves  = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    move_imb  = float(my_moves/ (my_moves + opp_moves))
    
    # maximise the imbalance in favour of current player and keep max possible distance from opponent
    return float(move_imb * distance)
    
    
def custom_score_2(game, player):
    """heuristic based on imbalance of players' move and tracking the opponent

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
    
    # get the Euclidean distance between players
    p_y, p_x = game.get_player_location(player)
    o_y, o_x = game.get_player_location(game.get_opponent(player))
    
    distance     = ((p_y - o_y)**2 + (p_x - o_x)**2) ** 0.5
    max_distance = (game.width ** 2 + game.height ** 2) ** 0.5

    # find move imbalance
    my_moves  = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    move_imb  = float(my_moves/ (my_moves + opp_moves))
                    
    # maximise the imbalance in favour of current player and track the opponent closely
    return float(move_imb + (1.0 - distance/ max_distance))
   

def custom_score_3(game, player):
    """Heuristic based on player's move imbalance and keeping optimal Euclidean distance from opponent
    move imbalance.
    
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
        
    # get the Euclidean distance between players
    p_y, p_x = game.get_player_location(player)
    o_y, o_x = game.get_player_location(game.get_opponent(player))
    distance = ((p_y - o_y)**2 + (p_x - o_x)**2) ** 0.5
    
    # find move imbalance
    my_moves  = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    move_imb  = float(my_moves/ (my_moves + opp_moves))
    
    # maximise the imbalance in favour of current player and keep max possible distance from opponent
    return float(move_imb * distance)


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth       = search_depth
        self.score              = score_fn
        self.time_left          = None
        self.TIMER_THRESHOLD    = timeout
        # to hold the information if current search has found the terminal state.
        self.terminal_state     = False
        # record the average search time
        self.branching_factor   = 8
        self.level_search_time  = 0.
        self.level_search_count = 0
        self.level_avg_time     = 0.
               

class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # set default move
        best_move = (-1, -1)
        
        # on timeout, return the next available move
        legal_moves = game.get_legal_moves()
        if len(legal_moves) > 0:
            best_move = legal_moves[0]

        try:
           
            # search the optimal move by minimax
            _move = self.minimax(game, self.search_depth)
            
            if _move != (-1, -1):
                best_move = _move

        except SearchTimeout:
           pass
        
        # return the best estimated move
        return best_move

        
    def minimax(self, game, depth):
        """Depth-limited minimax search algorithm implementation 
           Ref: https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md
        
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves
        """
        
        # find the optimal score and corresponding move
        _score, _move = self.search_tree(game, depth, True)
            
        return _move
    
        
    
    def search_tree(self, game, depth, max_player):
        """Search the game tree to find the optimal score and corresponding move by 
           exploring the tree upto specified depth levels
         
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        move: (int, int)
            The board coordinates to take the nxt move by current player
            
        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree from current level
            
        max_player: bool
            To indicate if the current game is with max player or with min player.
            True if active player is the max player
            False if active plater is the min player

        Returns
        -------
        (float, (int, int))
            A tuple of estimated optimal score and corresponding move

        """
        
        # underestimate the remaining time for stack unwinding
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        
        optimal_score = None
        optimal_move  = (-1, -1)
        player        = game.active_player
        
        # if min player is playing the game, max player is the inactive player
        if max_player == False:
            player = game.inactive_player
        
        
        # if this is the restricted terminal node, get the score for this node
        if(depth <= 0):
            optimal_score = self.score(game, player)
            return (optimal_score, optimal_move)
        
                
        # get the available legal moves for current player
        legal_moves = game.get_legal_moves()
        
        # if no more legal moves allowed, return current state
        if not legal_moves:
            optimal_score = self.score(game, player)
            return (optimal_score, optimal_move)
        
        # not in terminal state - expand the game tree for next level    
        
        if max_player:
            
            # move the game to min player now
            optimal_score = float("-inf")
            
            # expand tree for all available moves
            for next_move in legal_moves: 
                                
                # play this move and explore the sub tree
                next_game = game.forecast_move(next_move)
                
                _score, _move = self.search_tree(next_game, depth -1, False)
                
                if _score > optimal_score:
                    # better move found
                    optimal_score = _score
                    optimal_move = next_move
                    
                # if found a winning move, stop here
                if optimal_score == float("inf"):
                    break
        else:
            
            # move the game to max player now
            optimal_score = float("inf")
            
            # expand tree for all available moves
            for next_move in legal_moves:
                
                # play this move and explore the sub tree
                next_game = game.forecast_move(next_move)
                
                _score, _move = self.search_tree(next_game, depth -1, True)
                
                if _score < optimal_score:
                    # better move found
                    optimal_score = _score
                    optimal_move = next_move
                    
                # if found a winning move, stop here
                if optimal_score == float("-inf"):
                    break
         
        # return optimal score and corresponding move
        return (optimal_score, optimal_move)
        


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        
        self.time_left = time_left
        
        # reset terminal state indicator
        # This should be done using local variable instead of class variable
        # but due to restriction on alphabeta method interface, class variable is being used.
        self.terminal_state = False
        
        best_move = (-1, -1)
     
        try:
           
            # get the next best avaiable move
            best_move = self.alphabeta(game, 1, True)
             
            # now start the iterative deep search
            iterations = max(2, self.get_starting_iteration(self.time_left()))
                                    
            # Continue the iterative deep search as long as time permits or 
            # a terminal state has been found
            while self.time_left() > max(self.TIMER_THRESHOLD, self.level_avg_time):
                
                # CANOT USE TIMESTAMP DUE TO TIME MODULE IMPORT RESTRICTION
                #timestamp = time.time()
                
                _move = self.alphabeta(game, iterations, True)
                
                if _move != (-1, -1):
                    best_move = _move
                
                # update search estimate
                #self.level_search_time += (time.time() - timestamp)
                #self.level_search_count += iterations
                #self.level_avg_time = self.level_search_time / self.level_search_count
                
                # if we have found terminal state - stop search now
                if self.terminal_state:
                    break
                
                iterations += 1
                                         
        except SearchTimeout:
           pass
        
        return best_move

        

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Depth-limited minimax search with alpha-beta pruning 
        Ref: https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int,int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves       
        """
                                
        _score, _move = self.search_tree(game, depth, alpha, beta, True)
        
        # check if search has found the terminal state
        if _score == float("inf") or _score == float("-inf"):
            # a terminal state found. Iterative deep search can be stopped now
            self.terminal_state = True
        
        return _move
                   
        
    def search_tree(self, game, depth, alpha, beta, max_player):
        """Search the game tree to find the optimal score and corresponding move by exploring the game 
         tree upto specified depth levels using alpha-beta pruning
         
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        move: (int, int)
            The board coordinates to take the nxt move by current player
            
        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree from current level
            
        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers
            
        max_player: bool
            To indicate if the current game is with max player or with min player.
            True if active player is the max player
            False if active plater is the min player

        Returns
        -------
        (float, (int, int))
            A tuple of estimated optimal score and corresponding move
        """
        
        # underestimate the remaining time for stack unwinding
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        
        optimal_score = None
        optimal_move  = (-1, -1)
        player        = game.active_player
        
        # if min player is playing the game, max player is the inactive player
        if max_player == False:
            player = game.inactive_player
        
        
        # if this is the restricted terminal node, get the score for this node
        if(depth <= 0):
            optimal_score = self.score(game, player)
            return (optimal_score, optimal_move)
        
                
        # get the available legal moves for current player
        legal_moves = game.get_legal_moves()
        
        # if no more legal moves allowed, return current state
        if not legal_moves:
            optimal_score = self.score(game, player)
            return (optimal_score, optimal_move)
        
        
        # not in terminal state - expand the game tree for next level    
        
        if max_player:
            
            # move the game to min player now
            optimal_score = float("-inf")
            
            # expand tree for all available moves
            for next_move in legal_moves: 
            
                # play this move and explore the sub tree
                next_game = game.forecast_move(next_move)
                
                _score, _move = self.search_tree(next_game, depth -1, alpha, beta, False)
                
                if _score > optimal_score:
                    # better move found
                    optimal_score = _score
                    optimal_move = next_move
                
                # if current node has value over beta, prune this branch
                if optimal_score >= beta:
                    break
                
                # update alpha
                alpha = max(alpha, optimal_score)
                
        else:
            
            # move the game to max player now
            optimal_score = float("inf")
            
            for next_move in legal_moves:
                     
                # play this move and explore the sub tree
                next_game = game.forecast_move(next_move)
                
                _score, _move = self.search_tree(next_game, depth -1, alpha, beta, True)
                
                if _score < optimal_score:
                    # better move found
                    optimal_score = _score
                    optimal_move = next_move
                
                # if current node has a value below alpha, prune this branch
                if optimal_score <= alpha: 
                    break 
                
                # update beta
                beta = min(beta, optimal_score)
         
        # return optimal score and corresponding move 
        return (optimal_score, optimal_move)
        

    def get_starting_iteration(self, time_allowed):
        """Find the starting depth level for iterative deep search by based on 
        learned experience from naive search

        Parameters
        ----------
        time_allowed : float
            allowed time for the search in secods
            
        Returns
        -------
        int
            Starting iteration for depth search 
        """
        
        # make sure there is enough samples before finding samples
        if self.level_search_count < 5:
            return 2
        
        iterations = 2
                
        while self.level_avg_time * (self.branching_factor ** iterations) < 0.33 * time_allowed:
            iterations += 1
            
            if iterations >= 5:
                break
                              
        return iterations