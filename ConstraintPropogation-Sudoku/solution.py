assignments = []

rows = 'ABCDEFGHI'
cols = '123456789'

def assign_value(values, box, value):
    """
    Please use this function to update your values dictionary!
    Assigns a value to a given box. If it updates the board record it.
    """

    # Don't waste memory appending actions that don't actually change any values
    if values[box] == value:
        return values

    values[box] = value
    if len(value) == 1:
        assignments.append(values.copy())
    return values


def naked_twins(values):
    """Eliminate values using the naked twins strategy.
    Args:
        values(dict): a dictionary of the form {'box_name': '123456789', ...}

    Returns:
        the values dictionary with the naked twins eliminated from peers.
    """

    # Find all instances of naked twins per unit and eliminate those digits from other boxes in 
    # that unit as possibility.
        
    # scan all units- row, column, square and diagonal
    for unit in unitlist:
        # get boxes having exactly 2 options
        unfilled_boxes = [box for box in unit if len(values[box]) == 2]

        for box in unfilled_boxes:            
            digits = values[box]
  
            # find the box that has exactly same value as this box 
            twin_unit_peer = [peer for peer in unit if peer != box and values[peer] == digits]
     
            if len(twin_unit_peer) > 0:
                
                # found twins in the unit
                twin_peer = twin_unit_peer[0]
                twin_set = set(digits)

                # get all boxes in this unit who have these two digits as possibility except these twins
                match_peers = [peer for peer in unit \
                              if peer != twin_peer and peer != box \
                              and len(values[peer]) > 1 \
                              and len(set(values[peer]) - twin_set) < len(values[peer])]
                                  
                # remove these digits from matched peers
                for digit in digits:
                    for peer in match_peers: 
                        assign_value(values, peer, values[peer].replace(digit, ''))
                
    return values
    

def cross(A, B):
    "Cross product of elements in A and elements in B."
    return [a+b for a in A for b in B]

def grid_values(grid):
    """
    Convert grid into a dict of {square: char} with '123456789' for empties.
    Args:
        grid(string) - A grid in string form.
    Returns:
        A grid in dictionary form
            Keys: The boxes, e.g., 'A1'
            Values: The value in each box, e.g., '8'. If the box has no value, then the value will be '123456789'.
    """
    map = {}

    # create the sudoku map with filled boxes with their digits and empty boxes with possbile values
    for b,c in zip(boxes, grid):
        if c == '.':
            map[b] = '123456789'
        else:
            map[b] = c
    
    return map

    
def display(values):
    """
    Display the values as a 2-D grid.
    Args:
        values(dict): The sudoku in dictionary form
    """
    
    # show the sudoku box in 9x9 grid with separation of 3x3 squares
    width = 1 + max(len(values[s]) for s in boxes)
    
    line = '+'.join(['-' * (width * 3)] * 3)
    
    for r in rows:
        print(''.join(values[r + c].center(width) + ('|' if c in '36' else '')
                      for c in cols))
        if r in 'CF': print(line)
        
    return

    
def eliminate(values):
    
    '''
    Eliminate possible values using elimination strategy.
     Args:
        values(dict): a dictionary of the form {'box_name': '123456789', ...}

    Returns:
        the updated values dictionary after applying elimination 
    '''
    
    # filled boxes cannot be a possibility for their unfilled peers.
    # remove such digits from unfilled boxes.
    
    filled_boxes = [box for box in values.keys() if len(values[box]) == 1] 
    
    for b in filled_boxes: 
        # eliminate this option from all its peers
        digit = values[b] 
        for p in peers[b]:
            assign_value(values, p, values[p].replace(digit, ''))
    
    return values

    
def only_choice(values):
    '''
     Fill the box using only choice strategy
     Args:
        values(dict): a dictionary of the form {'box_name': '123456789', ...}

    Returns:
        the updated values dictionary after applying only choice strategy 
    '''
    
    # scan all units
    for unit in unitlist:
        for digit in '123456789':
            
            # check if current digit is only option for this box in its unit
            # if yes, we can assign this digit to current box
            choices = [box for box in unit if digit in values[box]]

            if(len(choices) == 1):
                assign_value(values, choices[0], digit)
                
    return values

    
def reduce_puzzle(values):
    
    '''
     Try to solve the puzzle using constraint propogation
     Args:
        values(dict): a dictionary of the form {'box_name': '123456789', ...}

    Returns:
        False- if constraint propogation results in invalid state (one or more boxes failed to find any digit)
        Othersie - updated values dictionary after applying constraint propogration
    '''
    
    # record the change in the state so that constraint propogation can be stopped once it 
    # stops changing the puzzle state
    has_update = True
    current_state = len([box for box in values.keys() if len(values[box]) == 1])
    
    # Reduce the puzzle by elimination and only choice replacement.
    while has_update is True:
        
        # try reducing the puzzle
        values = eliminate(values)
        values = naked_twins(values)
        values = only_choice(values)
                
        # check if state has changed
        new_state = len([box for box in values.keys() if len(values[box]) == 1])
        has_update = new_state > current_state
        current_state = new_state
                              
        # for invalid placements during search, box may not have any option
        if len([box for box in values.keys() if len(values[box]) == 0]):
            return False

    return values
        


def search(values):
           
    '''
     Try to solve the puzzle using search tree and constraint propogation
     Args:
        values(dict): a dictionary of the form {'box_name': '123456789', ...}

    Returns:
        False- if puzzle state goes into an invalid state (one or more boxes failed to find any digit)
        Othersie - updated values dictionary
    '''
    
    # reduce the puzzle using constrain propogation
    values = reduce_puzzle(values)
    if values == False:
        return False
    
    # check if there are any unfilled boxes - if not, stop here
    unfilled_boxes = [box for box in values.keys() if len(values[box]) > 1]
    if(len(unfilled_boxes) == 0):
        return values
    
    # puzzle is not solved yet - apply the search strategy
    # pick the first unfiled box with min spanning tree for search
    box = sorted(unfilled_boxes, key = lambda box: len(values[box]))[0]
    
    # apply DFS search for every possible digit
    for digit in values[box]:

        reduced_values = dict(values)
        
        assign_value(reduced_values, box, digit)
        
        solved_values = search(reduced_values)
        
        if solved_values: 
            return solved_values

def solve(grid):
    """
    Find the solution to a Sudoku grid.
    Args:
        grid(string): a string representing a sudoku grid.
            Example: '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    Returns:
        The dictionary representation of the final sudoku grid. False if no solution exists.
    """
    
    # make the sudoku box dictionary
    values = grid_values(grid)
                  
    # solve the puzzle
    solution = search(values)
            
    return solution
    


# make sudoku boxes, units and peers
boxes           = cross(rows, cols)
row_units       = [cross(r, cols) for r in rows]
col_units       = [cross(rows, c) for c in cols]
square_units    = [cross(row, col) for row in ('ABC', 'DEF', 'GHI') for col in ('123', '456', '789')]

# diagonal constraints
cross_unit1     = [[r+c for r,c in zip(rows, cols)]]
cross_unit2     = [[r+c for r,c in zip(rows, cols[::-1])]]

# combine all units
unitlist        = row_units + col_units + square_units + cross_unit1 + cross_unit2
units           = dict((s, [u for u in unitlist if s in u]) for s in boxes)
peers           = dict((s, set(sum(units[s], [])) - set([s])) for s in boxes)

    
if __name__ == '__main__':
    diag_sudoku_grid = '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    display(solve(diag_sudoku_grid))

    try:
        from visualize import visualize_assignments
        visualize_assignments(assignments)

    except SystemExit:
        pass
    except:
        print('We could not visualize your board due to a pygame issue. Not a problem! It is not a requirement.')
