WIDTH = HEIGHT = 512
DIMENSION = 8
CELL_SZ = WIDTH // DIMENSION
IMAGES = {}

STATE = [
  ['br', 'bn', 'bb', 'bq', 'bk', 'bb', 'bn', 'br'],
  ['bp', 'bp', 'bp', 'bp', 'bp', 'bp', 'bp', 'bp'],
  ['--', '--', '--', '--', '--', '--', '--', '--'],
  ['--', '--', '--', '--', '--', '--', '--', '--'],
  ['--', '--', '--', '--', '--', '--', '--', '--'],
  ['--', '--', '--', '--', '--', '--', '--', '--'],
  ['wp', 'wp', 'wp', 'wp', 'wp', 'wp', 'wp', 'wp'],
  ['wr', 'wn', 'wb', 'wq', 'wk', 'wb', 'wn', 'wr']
]

PIECESCORE = {'k': 0, 'q': 9, 'r': 5, 'n': 3, 'b': 3, 'p': 1}

KNIGHT_SCORE = [
    [-5, -4, -3, -3, -3, -3, -4, -5],
    [-4, -2,  0,  0,  0,  0, -2, -4],
    [-3,  0,  1,  2,  2,  1,  0, -3],
    [-3,  0,  2,  3,  3,  2,  0, -3],
    [-3,  0,  2,  3,  3,  2,  0, -3],
    [-3,  0,  1,  2,  2,  1,  0, -3],
    [-4, -2,  0,  0,  0,  0, -2, -4],
    [-5, -4, -3, -3, -3, -3, -4, -5]
]

BISHOP_SCORE = [
    [-2, -1, -1, -1, -1, -1, -1, -2],
    [-1,  0,  0,  0,  0,  0,  0, -1],
    [-1,  0,  1,  2,  2,  1,  0, -1],
    [-1,  1,  1,  2,  2,  1,  1, -1],
    [-1,  0,  2,  2,  2,  2,  0, -1],
    [-1,  1,  1,  1,  1,  1,  1, -1],
    [-1,  0,  0,  0,  0,  0,  0, -1],
    [-2, -1, -1, -1, -1, -1, -1, -2]
]

QUEEN_SCORE = [
    [-2, -1, -1, -0.5, -0.5, -1, -1, -2],
    [-1,  0,  0,  0,   0,   0,  0, -1],
    [-1,  0,  0.5, 0.5, 0.5, 0.5, 0, -1],
    [-0.5, 0, 0.5, 1,   1,   0.5, 0, -0.5],
    [-0.5, 0, 0.5, 1,   1,   0.5, 0, -0.5],
    [-1,  0,  0.5, 0.5, 0.5, 0.5, 0, -1],
    [-1,  0,  0,   0,   0,   0,  0, -1],
    [-2, -1, -1, -0.5, -0.5, -1, -1, -2]
]

ROOK_SCORE = [
    [0,  0,  0,  0,  0,  0,  0,  0],
    [0.5, 1,  1,  1,  1,  1,  1, 0.5],
    [-0.5, 0,  0,  0,  0,  0,  0, -0.5],
    [-0.5, 0,  0,  0,  0,  0,  0, -0.5],
    [-0.5, 0,  0,  0,  0,  0,  0, -0.5],
    [-0.5, 0,  0,  0,  0,  0,  0, -0.5],
    [-0.5, 0,  0,  0,  0,  0,  0, -0.5],
    [0,  0,  0,  0.5, 0.5,  0,  0,  0]
]

WHITE_PAWN_SCORE = [
    [3,  3,  3,  3,  3,  3,  3,  3],
    [5,  5,  5,  5,  5,  5,  5,  5],
    [1,  1,  2,  3,  3,  2,  1,  1],
    [0.5, 0.5, 1,  2.5, 2.5, 1, 0.5, 0.5],
    [0,  0,  0,  2,  2,  0,  0,  0],
    [0.5, -0.5, -1, 0,  0, -1, -0.5, 0.5],
    [0.5,  1,  1, -2, -2,  1,  1,  0.5],
    [0,  0,  0,  0,  0,  0,  0,  0]
]

BLACK_PAWN_SCORE = [
    [0,  0,  0,  0,  0,  0,  0,  0],
    [0.5,  1,  1, -2, -2,  1,  1,  0.5],
    [0.5, -0.5, -1, 0,  0, -1, -0.5, 0.5],
    [0,  0,  0,  2,  2,  0,  0,  0],
    [0.5, 0.5, 1,  2.5, 2.5, 1, 0.5, 0.5],
    [1,  1,  2,  3,  3,  2,  1,  1],
    [5,  5,  5,  5,  5,  5,  5,  5],
    [3,  3,  3,  3,  3,  3,  3,  3]
]

PIECE_POSITIONS_SCORE = {'n':KNIGHT_SCORE,'b':BISHOP_SCORE,'q':QUEEN_SCORE,'r':ROOK_SCORE,'bp':BLACK_PAWN_SCORE,'wp':WHITE_PAWN_SCORE}
WEIGHT_SCORE = {'n': 0.5, 'b': 0.5, 'q': 0.75, 'r': 0.5, 'p': 0.25}
CENTER_CONTROL_BONUS = 0.5
CHECK_BONUS = 5
CHECKMATE = 10000000
KING_SAFETY_PENALTY = -1
MOBILITY_BONUS = 0.3

DIFFICULTY_LEVELS = {
    'easy': 3, 
    'medium': 5, 
    'hard': 7    
}
