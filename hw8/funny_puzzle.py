import heapq
from itertools import permutations

def get_manhattan_distance(from_state, to_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: implement this function. This function will not be tested directly by the grader. 

    INPUT: 
        Two states (if second state is omitted then it is assumed that it is the goal state)

    RETURNS:
        A scalar that is the sum of Manhattan distances for all tiles.
    """
    # manhattan([x1,y1], [x2,y2]) = abs(x1-x2) + abs(y1-y2)
    distance = 0
    for tile in range(1, 8):
        i = from_state.index(tile)
        j = to_state.index(tile)
        x1, y1 = i // 3, i % 3
        x2, y2 = j // 3, j % 3
        distance += abs(x1-x2) + abs(y1-y2)
    return round(distance, 3)

    
def get_euclidean_distance(from_state, to_state=(1,2,3,4,5,6,7,0,0)):
    """
    TODO: Implement this function. This function will not directly be tested by the grader.
    INPUT: 
        Two states (if second state is omitted then it is assumed that it is the goal state)

    RETURNS:
        A scalar that is the sum of Euclidean distances for all tiles.
    """
    # euclidean([x1,y1], [x2,y2]) = sqrt((x1-x2)^2 + (y1-y2)^2)
    distance = 0
    for tile in range(1, 8):
        i = from_state.index(tile)
        j = to_state.index(tile)
        x1, y1 = i // 3, i % 3
        x2, y2 = j // 3, j % 3
        distance += ((x1-x2)**2 + (y1-y2)**2)**0.5
    return round(distance, 3)


def print_succ(state,heuristic):
    """
    TODO: This is based on get_succ function below, so should implement that function.

    INPUT: 
        A state (list of length 9)
        The hueristic to use ("manhattan" or "euclidean")

    WHAT IT DOES:
        Prints the list of all the valid successors in the puzzle. 
    """
    succ_states = get_succ(state,heuristic)

    for succ_state in succ_states:
        if heuristic == "manhattan":
            print(succ_state, "h={}".format(get_manhattan_distance(succ_state)))
        else:
            print(succ_state, "h={}".format(get_euclidean_distance(succ_state)))


def get_succ(state,heuristic):
    """
    TODO: implement this function.

    INPUT: 
        An initial state (list of length 9)
        The hueristic to use ("manhattan" or "euclidean")

    RETURNS:
        A list of all the valid successors in the puzzle (don't forget to sort the result as done below). 
    """
    succ_states = []
    directions = [(-1,0),(1,0),(0,-1),(0,1)]  # up, down, left, right

    zero_indices = [i for i, v in enumerate(state) if v == 0]

    for zi in zero_indices:
        x, y = zi // 3, zi % 3

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            if 0 <= nx < 3 and 0 <= ny < 3:
                ni = nx * 3 + ny

                if state[ni] != 0:  # must swap with tile
                    new_state = state.copy()
                    new_state[zi], new_state[ni] = new_state[ni], new_state[zi]

                    if new_state not in succ_states:
                        succ_states.append(new_state)

    return sorted(succ_states)
    

def solve(state, heuristic, silent, goal_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: Implement the A* algorithm here.

    INPUT: 
        An initial state (list of length 9)
        The hueristic to use ("manhattan" or "euclidean")
        silent, which is False if the results are to be printed, and True if nothing should be printed

    WHAT IT SHOULD DO:
        Prints a path of configurations from initial state to goal state along  h values, number of moves, and max queue number in the format specified in the pdf.
    """
    OPEN = []
    CLOSED = set()
    g_best = {}
    parent = {}
    nodes = {}
    def h_fn(s):
        if heuristic == "manhattan":
            return get_manhattan_distance(s)
        else:
            return get_euclidean_distance(s)
        
    # 1. Put the start state S on the priority queue. We call the priority queue OPEN
    g0 = 0
    h0 = h_fn(state)
    f0 = g0 + h0

    heapq.heappush(OPEN, (f0, state))

    g_best[tuple(state)] = 0
    parent[tuple(state)] = None
    nodes[tuple(state)] = (state, h0, 0)

    max_length = 1
    nodes_expanded = 0
    goal = None
    
    # 2. If OPEN is empty, exit with failure
    while OPEN:
        # 3. Remove from OPEN and place on CLOSED a node n for which f(n) is minimum
        cost, curr_state = heapq.heappop(OPEN)
        nodes_expanded += 1

        if tuple(curr_state) in CLOSED:
            continue
        CLOSED.add(tuple(curr_state))
        
        # 4. If n is a goal node, exit (recover path by tracing back pointers from n to S)
        if curr_state == goal_state:
            goal = curr_state
            break
        g_n = g_best[tuple(curr_state)]

        # 5. Expand n, generating all successors and attach to pointers back to n. 
        for succ in get_succ(curr_state, heuristic):
            # For each successor n' of n
            g_new = g_n + 1
            h_new = h_fn(succ)
            f_new = g_new + h_new

            t = tuple(succ)

            # 1. If n' is not already on OPEN or CLOSED compute h(n'), g(n')=g(n)+ c(n,n'), f(n')=g(n')+h(n’), and place it on OPEN.
            if t not in g_best and t not in CLOSED:

                heapq.heappush(OPEN, (f_new, succ))
                g_best[t] = g_new
                parent[t] = curr_state

                nodes[t] = (succ, h_new, g_new)
            # If n' is already on OPEN or CLOSED, then check if g(n') is lower for the new version of n’.
            else:
                if g_new < g_best.get(t, float('inf')):
                    # If so, then:
                    # 1. Redirect pointers backward from n' along path yielding lower g(n').
                    g_best[t] = g_new
                    parent[t] = curr_state
                    # 2. If (n’ is already on OPEN) then update n' on OPEN; else add n' to OPEN
                    nodes[t] = (succ, h_new, g_new)
                    heapq.heappush(OPEN, (f_new, succ))
                    # 3. If g(n') is not lower for the new version, do nothing. 
        
        max_length = max(max_length, len(OPEN))

    state_info_list = []
    cur = goal

    while cur is not None:
        t = tuple(cur)
        state_val, h_val, g_val = nodes[t]
        state_info_list.append((state_val, h_val, g_val))
        cur = parent[t]

    state_info_list.reverse()

    # This is a format helper，which is only designed for format purpose.
    # build "state_info_list", for each "state_info" in the list, it contains "current_state", "h" and "move".
    # define and compute max length
    # it can help to avoid any potential format issue.
    if not silent:
        for state_info in state_info_list:
            current_state = state_info[0]
            h = state_info[1]
            move = state_info[2]
            print(current_state, "h={}".format(h), "moves: {}".format(move))
        print("Max queue length: {}".format(max_length))

    return max_length, nodes_expanded

def compare():
    """
    TODO: implement this function.

    WHAT IT DOES:
        Prints the average max queue length and average nodes expanded for A* with
        Manhattan and Euclidean heuristics across all 24 puzzles where the first 5
        values are [0, 0, 7, 6, 5]. Also includes a written explanation of which
        heuristic is more efficient and why.
    """

    prefix = [0, 0, 7, 6, 5]
    remaining = [1, 2, 3, 4]
    all_puzzles = [prefix + list(perm) for perm in permutations(remaining)]

    # TODO: solve each puzzle with both heuristics and collect results
    # hint: use the solve function with silent=True
    m_total_queue = 0
    e_total_queue = 0
    m_total_expanded = 0
    e_total_expanded = 0

    for puzzle in all_puzzles:
        m_q, m_e = solve(puzzle, "manhattan", True)
        e_q, e_e = solve(puzzle, "euclidean", True)

        m_total_queue += m_q
        e_total_queue += e_q
        m_total_expanded += m_e
        e_total_expanded += e_e

    n = len(all_puzzles)

    m_max_queue = m_total_queue / n
    e_max_queue = e_total_queue / n
    m_expanded = m_total_expanded / n
    e_expanded = e_total_expanded / n

    print("Manhattan Average Max Queue Length: {:.2f}".format(m_max_queue))
    print("Euclidean Average Max Queue Length: {:.2f}".format(e_max_queue))
    print("Manhattan Average Expanded: {:.2f}".format(m_expanded))
    print("Euclidean Average Expanded: {:.2f}".format(e_expanded))
    
    # TODO: print written explanation
    print("Manhattan is better because it avoids squaring and is more robust than Euclidean distance.")


if __name__ == "__main__":
    """
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    """
    print_succ([3, 4, 6, 0, 0, 1, 7, 2, 5], 'manhattan')
    print()
    print_succ([3, 4, 6, 0, 0, 1, 7, 2, 5], 'euclidean')
    print()
    # print_succ([6, 0, 0, 3, 5, 1, 7, 2, 4], 'manhattan')
    # print_succ([0, 4, 7, 1, 3, 0, 6, 2, 5], 'manhattan')
    # print_succ([5, 2, 3, 0, 6, 4, 7, 1, 0], 'euclidean')
    # print_succ([1, 7, 0, 6, 3, 2, 0, 4, 5], 'euclidean')
    solve([3, 4, 6, 0, 0, 1, 7, 2, 5],'manhattan',False)
    solve([3, 4, 6, 0, 0, 1, 7, 2, 5],'euclidean',False)
    # solve([6, 0, 0, 3, 5, 1, 7, 2, 4],'manhattan',False)
    # solve([0, 4, 7, 1, 3, 0, 6, 2, 5],'manhattan',False)
    #solve([5, 2, 3, 0, 6, 4, 7, 1, 0],'euclidean',False)
    # solve([1, 7, 0, 6, 3, 2, 0, 4, 5],'euclidean',False)
    compare()

