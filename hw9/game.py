import copy
import random

class TeekoPlayer:
    """ An object representation for an AI game player for the game Teeko.
    """
    pieces = ['b', 'r']
    max_depth = 2

    def __init__(self):
        """ Initializes a TeekoPlayer object by randomly selecting red or black as its
        piece color.
        """
        self.board = [[' ' for j in range(5)] for i in range(5)]
        self.my_piece = random.choice(self.pieces)
        self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1]

    def make_move(self, state):
        """ 
        TODO: Selects a (row, col) space for the next move. You may assume that whenever
        this function is called, it is this player's turn to move.

        Args:
            state (list of lists): should be the current state of the game as saved in
                this TeekoPlayer object. Note that this is NOT assumed to be a copy of
                the game state and should NOT be modified within this method (use
                place_piece() instead). Any modifications (e.g. to generate successors)
                should be done on a deep copy of the state.

                In the "drop phase", the state will contain less than 8 elements which
                are not ' ' (a single space character).

        Return:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

        Note that without drop phase behavior, the AI will just keep placing new markers
            and will eventually take over the board. This is not a valid strategy and
            will earn you no points.
        """
        
        best_score = float('-inf')
        best_state = None

        for s in self.succ(state, self.my_piece):
            score = self.min_value(s, 1)
            if score > best_score:
                best_score = score
                best_state = s

        # extract move from state difference
        move = []

        if self.is_drop_phase(state):
            for i in range(5):
                for j in range(5):
                    if state[i][j] != best_state[i][j]:
                        return [(i, j)]
        else:
            src = None
            dst = None
            for i in range(5):
                for j in range(5):
                    if state[i][j] != best_state[i][j]:
                        if state[i][j] == self.my_piece:
                            src = (i, j)
                        elif best_state[i][j] == self.my_piece:
                            dst = (i, j)

            return [dst, src]

        return move

    def succ(self, state, my_piece): 
        """
        TODO: Generate a list of valid successors for the current game state 
        on placing your piece. (defined by self.my_piece)
        """
        successors = []

        if self.is_drop_phase(state):    
            has_piece = any(cell != ' ' for row in state for cell in row)
            successors = []
            for i in range(5):
                for j in range(5):
                    if state[i][j] == ' ':
                        if not has_piece or self.has_neighbor(state, i, j):
                            new_state = [row[:] for row in state]
                            new_state[i][j] = my_piece
                            successors.append(new_state)
            return successors
        # move phase
        else:
            for i in range(5):
                for j in range(5):
                    if state[i][j] == my_piece:
                        for di in [-1, 0, 1]:
                            for dj in [-1, 0, 1]:
                                if di == 0 and dj == 0:
                                    continue
                                ni, nj = i + di, j + dj
                                if 0 <= ni < 5 and 0 <= nj < 5:
                                    if state[ni][nj] == ' ':
                                        new_state = [row[:] for row in state]
                                        new_state[i][j] = ' '
                                        new_state[ni][nj] = my_piece
                                        successors.append(new_state)
            return successors[:8]
    
    def opponent_move(self, move):
        """ Validates the opponent's next move against the internal board representation.
        You don't need to touch this code.

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.
        """
        # validate input
        if len(move) > 1:
            source_row = move[1][0]
            source_col = move[1][1]
            if source_row != None and self.board[source_row][source_col] != self.opp:
                self.print_board()
                print(move)
                raise Exception("You don't have a piece there!")
            if abs(source_row - move[0][0]) > 1 or abs(source_col - move[0][1]) > 1:
                self.print_board()
                print(move)
                raise Exception('Illegal move: Can only move to an adjacent space')
        if self.board[move[0][0]][move[0][1]] != ' ':
            raise Exception("Illegal move detected")
        # make move
        self.place_piece(move, self.opp)

    def place_piece(self, move, piece):
        """ Modifies the board representation using the specified move and piece

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

                This argument is assumed to have been validated before this method
                is called.
            piece (str): the piece ('b' or 'r') to place on the board
        """
        if len(move) > 1:
            self.board[move[1][0]][move[1][1]] = ' '
        self.board[move[0][0]][move[0][1]] = piece

    def print_board(self):
        """ Formatted printing for the board """
        for row in range(len(self.board)):
            line = str(row)+": "
            for cell in self.board[row]:
                line += cell + " "
            print(line)
        print("   A B C D E")

    
    def heuristic_game_value(self, state):
        """ 
        TODO: Define the heuristic game value of the current board state taking into account players
        and opponents

        Args:
        state (list of lists): either the current state of the game as saved in
            this TeekoPlayer object, or a generated successor state.

        Returns:
            float heuristic_val (heuristic computed for the game state)
        """
        val = self.game_value(state)
        if val != 0:
            return val

        def score_line(line):
            if self.opp in line and self.my_piece in line:
                return 0  # blocked
            count_my = line.count(self.my_piece)
            count_opp = line.count(self.opp)
            if count_my > 0:
                return count_my ** 2   # reward bigger chains more
            if count_opp > 0:
                return -(count_opp ** 2)
            return 0

        score = 0

        # reuse same windows as game_value
        lines = []

        for i in range(5):
            for j in range(2):
                lines.append([state[i][j+k] for k in range(4)])
                lines.append([state[j+k][i] for k in range(4)])

        for i in range(2):
            for j in range(2):
                lines.append([state[i+k][j+k] for k in range(4)])
                lines.append([state[i+3-k][j+k] for k in range(4)])

        for i in range(4):
            for j in range(4):
                lines.append([state[i][j], state[i][j+1],
                            state[i+1][j], state[i+1][j+1]])

        for line in lines:
            score += score_line(line)

        return score / 10.0  # normalize to (-1,1)
 
    def game_value(self, state):
        """ 
        TODO: Checks the current board status for a win condition

        Args:
        state (list of lists): either the current state of the game as saved in
            this TeekoPlayer object, or a generated successor state.

        Returns:
            int: 1 if this TeekoPlayer wins, -1 if the opponent wins, 0 if no winner
        """
        lines = []

        # horizontal & vertical
        for i in range(5):
            for j in range(2):
                lines.append([state[i][j+k] for k in range(4)])  # horizontal
                lines.append([state[j+k][i] for k in range(4)])  # vertical

        # diagonals
        for i in range(2):
            for j in range(2):
                lines.append([state[i+k][j+k] for k in range(4)])  # \
                lines.append([state[i+3-k][j+k] for k in range(4)])  # /

        # 2x2 boxes
        for i in range(4):
            for j in range(4):
                box = [state[i][j], state[i][j+1],
                    state[i+1][j], state[i+1][j+1]]
                lines.append(box)

        for line in lines:
            if line.count(self.my_piece) == len(line):
                return 1
            if line.count(self.opp) == len(line):
                return -1

        return 0 # no winner yet
    
    def max_value(self, state, depth):
        """
        TODO: Complete the helper function to implement min-max as described in the writeup
        """
        return self._max_value_ab(state, depth, float('-inf'), float('inf'))

    def min_value(self, state, depth):
        return self._min_value_ab(state, depth, float('-inf'), float('inf'))

    def _max_value_ab(self, state, depth, alpha, beta):
        val = self.game_value(state)

        # terminal state
        if val != 0:
            return val

        # depth cutoff
        if depth >= self.max_depth:
            return self.heuristic_game_value(state)

        for s in self.succ(state, self.my_piece):
            alpha = max(alpha, self._min_value_ab(s, depth + 1, alpha, beta))
            # alpha cutoff
            if alpha >= beta:
                return beta
        return alpha

    def _min_value_ab(self, state, depth, alpha, beta):
        val = self.game_value(state)

        # terminal state
        if val != 0:
            return val

        # depth cutoff
        if depth >= self.max_depth:
            return self.heuristic_game_value(state)

        for s in self.succ(state, self.opp):
            beta = min(beta, self._max_value_ab(s, depth + 1, alpha, beta))
            # beta cutoff
            if alpha >= beta:
                return alpha
        return beta
    
    def is_drop_phase(self, state):
        count = 0
        for row in state:
            for cell in row:
                if cell != ' ':
                    count += 1
        return count < 8
    
    def has_neighbor(self, state, i, j):
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                ni, nj = i + di, j + dj
                if 0 <= ni < 5 and 0 <= nj < 5:
                    if state[ni][nj] != ' ':
                        return True
        return False


############################################################################
#
# THE FOLLOWING CODE IS FOR SAMPLE GAMEPLAY ONLY
#
############################################################################
def main():
    print('Hello, this is Samaritan')
    ai = TeekoPlayer()
    piece_count = 0
    turn = 0

    # drop phase
    while piece_count < 8 and ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved at "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                player_move = input("Move (e.g. B3): ")
                while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                    player_move = input("Move (e.g. B3): ")
                try:
                    ai.opponent_move([(int(player_move[1]), ord(player_move[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        piece_count += 1
        turn += 1
        turn %= 2

    # move phase - can't have a winner until all 8 pieces are on the board
    while ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved from "+chr(move[1][1]+ord("A"))+str(move[1][0]))
            print("  to "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                move_from = input("Move from (e.g. B3): ")
                while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                    move_from = input("Move from (e.g. B3): ")
                move_to = input("Move to (e.g. B3): ")
                while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                    move_to = input("Move to (e.g. B3): ")
                try:
                    ai.opponent_move([(int(move_to[1]), ord(move_to[0])-ord("A")),
                                    (int(move_from[1]), ord(move_from[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        turn += 1
        turn %= 2

    ai.print_board()
    if ai.game_value(ai.board) == 1:
        print("AI wins! Game over.")
    else:
        print("You win! Game over.")


#if __name__ == "__main__":
#    main()
