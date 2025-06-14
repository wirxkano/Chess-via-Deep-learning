import numpy as np
import chess
import chess.pgn
import io
from Game.Board import Board
from configs import *

def chess_label(pos):
    row, col = pos
    letters = 'abcdefgh'
    numbers = '12345678'
    column_label = letters[col]
    row_label = numbers[7 - row]
    return f"{column_label}{row_label}"

def algebraic_to_index(square):
    file_to_x = {'a': 0, 'b': 1, 'c': 2, 'd': 3,
                 'e': 4, 'f': 5, 'g': 6, 'h': 7}
    x = 8 - int(square[1])
    y = file_to_x[square[0]]
    return (x, y)

def label_to_move(label):
    label = str(label)
    from_square = label[:2]
    to_square = label[2:]
    from_pos = algebraic_to_index(from_square)
    to_pos = algebraic_to_index(to_square)
    return from_pos, to_pos

def add_castling_move(board, selected, current_player, moves):
    if board.state[selected[0]][selected[1]][1] == 'k':
      if board.can_castling('kingside', current_player):
        if current_player == 'w':
            moves.append((7, 6))
        else:
            moves.append((0, 6))
      if board.can_castling('queenside', current_player):
        if current_player == 'w':
            moves.append((7, 2))
        else:
            moves.append((0,2))
            
    return moves

# ============================== FOR DEEP LEARNING MODEL =======================================
# def mapping_piece(piece):
#     piece_color = 6 if piece[0] == 'w' else 0
#     piece_type = piece[1]
#     if piece_type == 'p':
#         piece_type = 0
#     elif piece_type == 'n':
#         piece_type = 1
#     elif piece_type == 'b':
#         piece_type = 2
#     elif piece_type == 'r':
#         piece_type = 3
#     elif piece_type == 'q':
#         piece_type = 4
#     elif piece_type == 'k':
#         piece_type = 5
        
#     return piece_color, piece_type
    
# def board_to_matrix(board):
#     matrix = np.zeros((13, 8, 8))

#     for r in range(DIMENSION):
#         for c in range(DIMENSION):
#             piece = board.state[r][c]
#             if piece != '--':
#                 piece_type, piece_color = mapping_piece(piece)
#                 matrix[piece_type + piece_color, r, c] = 1
                
#     valid_moves = board.get_all_moves(board.turn)
#     for move in valid_moves:
#         # to row and to column
#         matrix[12, move[1][0], move[1][1]] = 1
        
#     return matrix

# def preprocess(pgn_string):
#     X = []
#     y = []
#     game = chess.pgn.read_game(io.StringIO(pgn_string))
#     board = Board('w')
#     for move in game.mainline_moves():
#         X.append(board_to_matrix(board))
#         y.append(move.uci())
#         from_pos, to_pos = label_to_move(move)
#         board.make_move(from_pos, to_pos, False, False)
        
#     return X, y
        
# def encode_moves(moves, move_to_int):
#     return np.array([move_to_int[move] for move in moves])

import numpy as np


def board_to_matrix(board):
    # 8x8 is a size of the chess board.
    # 12 = number of unique pieces.
    # 13th board for legal moves (WHERE we can move)
    # maybe 14th for squares FROM WHICH we can move? idk
    matrix = np.zeros((13, 8, 8))
    piece_map = board.piece_map()

    # Populate first 12 8x8 boards (where pieces are)
    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        piece_type = piece.piece_type - 1
        piece_color = 0 if piece.color else 6
        matrix[piece_type + piece_color, row, col] = 1

    # Populate the legal moves board (13th 8x8 board)
    legal_moves = board.legal_moves
    for move in legal_moves:
        to_square = move.to_square
        row_to, col_to = divmod(to_square, 8)
        matrix[12, row_to, col_to] = 1

    return matrix


def create_input_for_nn(games):
    X = []
    y = []
    for game in games:
        board = game.board()
        for move in game.mainline_moves():
            X.append(board_to_matrix(board))
            y.append(move.uci())
            board.push(move)
    return np.array(X, dtype=np.float32), np.array(y)


def encode_moves(moves):
    move_to_int = {move: idx for idx, move in enumerate(set(moves))}
    return np.array([move_to_int[move] for move in moves], dtype=np.float32), move_to_int
        