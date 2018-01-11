import random

class SearchTimeout(Exception):
    pass

def custom_score(game, player):
    if game.is_loser(player):
        return float("-inf")
    
    if game.is_winner(player):
        return float("inf")

    opponent_player = game.get_opponent(player)
    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(opponent_player)
    
    own_moves_len = len(own_moves)
    for move in own_moves:
        next_game = game.forecast_move(move)
        own_moves_len += len(next_game.get_legal_moves(player))

    opp_moves_len = len(opp_moves)
    for move in opp_moves:
        next_game = game.forecast_move(move)
        opp_moves_len += len(next_game.get_legal_moves(opponent_player))

    return float(own_moves_len - opp_moves_len)

def custom_score_2(game, player):
    if game.is_loser(player):
        return float("-inf")
    
    if game.is_winner(player):
        return float("inf")

    opponent_player = game.get_opponent(player)
    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(opponent_player)
    own_moves_len = len(own_moves)
    opp_moves_len = len(opp_moves)
    return float(own_moves_len - opp_moves_len)

def custom_score_3(game, player):
    if game.is_loser(player):
        return float("-inf")
    
    if game.is_winner(player):
        return float("inf")

    opponent_player = game.get_opponent(player)
    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(opponent_player)
    own_moves_len = len(own_moves)
    opp_moves_len = len(opp_moves)
    return float(own_moves_len - 2*opp_moves_len)


class IsolationPlayer:
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

class MinimaxPlayer(IsolationPlayer):
    def get_move(self, game, time_left):
        self.time_left = time_left
        best_move = (-1, -1)
        try:
            return self.minimax(game, self.search_depth)
        except SearchTimeout:
            pass
        return best_move

    def min_value(self, game, depth_left):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        legal_moves = game.get_legal_moves()
        if len(legal_moves)==0:
            return 1
        if depth_left==0:
            return self.score(game, self)
        v = float("inf")
        for move in legal_moves:
            v = min(v, self.max_value(game.forecast_move(move), depth_left-1))
        return v

    def max_value(self, game, depth_left):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        legal_moves = game.get_legal_moves()
        if len(legal_moves)==0:
            return -1
        if depth_left==0:
            return self.score(game, self)
        v=float("-inf")
        for move in legal_moves:
            v = max(v, self.min_value(game.forecast_move(move), depth_left-1))
        return v
    
    def minimax(self, game, depth):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        moves = game.get_legal_moves()
        if len(moves)==0:
            return (-1,-1)
        else:
            result = max(moves, key=lambda m:self.min_value(game.forecast_move(m), depth-1))
        return result

class AlphaBetaPlayer(IsolationPlayer):
    def get_move(self, game, time_left):
        self.time_left = time_left
        legal_moves = game.get_legal_moves()
        if len(legal_moves)==0:
            best_move = (-1,-1)
        else:
            best_move = legal_moves[0]
        try:
            depth = 1
            while True:
                if self.time_left() < self.TIMER_THRESHOLD:
                    raise SearchTimeout()
                best_move = self.alphabeta(game, depth)
                depth += 1
        except SearchTimeout:
            pass
        return best_move
    
    def min_value(self, game, depth_left, alpha, beta):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
            
        legal_moves = game.get_legal_moves()
        
        if depth_left==0 or len(legal_moves)==0:
            return self.score(game, self)
        
        v = float("inf")
        for move in legal_moves:
            v = min(v, self.max_value(game.forecast_move(move), depth_left-1, alpha, beta))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    def max_value(self, game, depth_left, alpha, beta):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
            
        legal_moves = game.get_legal_moves()
        
        if depth_left==0 or len(legal_moves)==0:
            return self.score(game, self)
        
        v=float("-inf")
        for move in legal_moves:
            v = max(v, self.min_value(game.forecast_move(move), depth_left-1, alpha, beta)) 
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
            
        legal_moves = game.get_legal_moves()
        
        if len(legal_moves)==0:
            return (-1,-1)
        else:
            best_move = legal_moves[0]
            best_score = float('-Inf')
            for move in legal_moves:
                value = self.min_value(game.forecast_move(move), depth-1, alpha, beta)
                if value > best_score:
                    best_score = value
                    best_move = move
                alpha = max(alpha, best_score)
            return best_move