import copy
import pygame as pg
from chess import Board as PyBoard
from Game.Gui import Gui
from Game.Board import Board
from Game.Engine import Engine
from Minimax.Minimax import Minimax
from Random.RandomAgent import RandomAgent
from DeepLearning.predict import DeepLearningAgent
from Utils.auxiliary import *
from options import *
from configs import *

def main():
    args = parse_arguments()
    mode = args.mode
    first_player = args.first
    difficulty = args.difficulty.lower()
    
    if difficulty not in DIFFICULTY_LEVELS:
        print("Invalid difficulty level. Choose: easy, medium, hard.")
        return
    depth = DIFFICULTY_LEVELS[difficulty]

    pg.init()
    screen = pg.display.set_mode((WIDTH, HEIGHT))
    pg.display.set_caption("Chess")
    clock = pg.time.Clock()
    screen.fill("black")
    font = pg.font.Font('freesansbold.ttf', 12)

    current_player = first_player
    selected = None
    running = True

    gui = Gui()
    gui.load_images()  # Load images once after Gui is created
    board = Board(current_player)
    pyboard = PyBoard()
    state = STATE
    
    if mode == 'ava':
        # white_engine = Engine(board, Minimax(), depth=int(depth))
        white_engine = Engine(pyboard, DeepLearningAgent())
        black_engine = Engine(board, RandomAgent())
        # black_engine = Engine(board, Minimax(), depth=int(1))
    else:
        engine = Engine(DeepLearningAgent())

    valid_moves = []  # To store valid moves for highlighting

    while running:
        # GAME OVER
        if board.gameOver:
            print(board.gameOver)
            ip = input("Press Q to quit\n")
            if ip.lower() == "q":
                running = False
                break
        
        # AGENT MOVE
        if mode == 'ava' or (mode == 'pva' and current_player == 'b'):
            virtual_board = copy.deepcopy(board)
            
            if mode == 'ava':
                agent = white_engine if current_player == 'w' else black_engine
            else:
                agent = engine
            
            agent_move = agent.move(virtual_board, current_player)
            
            if agent_move is None:
                if board.gameOver:
                    print(board.gameOver)
                    ip = input("Press Q to quit\n")
                    if ip.lower() == "q":
                        running = False
                print(board.gameOver)
                break
            from_label, to_label = chess_label(agent_move[0]), chess_label(agent_move[1])
            uci_move = chess.Move.from_uci(f"{from_label}{to_label}")
            
            if pyboard.is_kingside_castling(uci_move) or pyboard.is_queenside_castling(uci_move):
                # print("Kingside castling – update UI accordingly")
                state = board.castling(agent_move[0], agent_move[1])
            else:
                state = board.make_move(agent_move[0], agent_move[1])
                
            row, col = agent_move[1]
            # Handle pawn promotion for agent
            state = board.promotion(row, col, current_player, player_choice=False)
            
            pyboard.push(uci_move)
            
            print(f'Agent {current_player} performs move from {from_label} to {to_label}')
            
            current_player = 'w' if current_player == 'b' else 'b'
            pg.time.delay(1000)
                
        # PERSON MOVE
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False

            elif event.type == pg.MOUSEBUTTONDOWN:
                pos = pg.mouse.get_pos()
                row, col = pos[1] // CELL_SZ, pos[0] // CELL_SZ

                if selected:
                    if (row, col) in valid_moves and not board.in_check_after_move(selected, (row, col), current_player):
                        if board.state[selected[0]][selected[1]][1] == 'k' and\
                            ((row, col) == (7,6) or (row, col) == (0,6) or (row, col) == (7,2) or (row, col) == (0,2)):
                            state = board.castling(selected, (row, col))
                        else:
                            state = board.make_move(selected, (row, col))
                        piece = board.state[row][col]
                        # Handle pawn promotion via GUI for human player
                        if piece[1] == 'p' and (row == 0 or row == 7):
                            choice = gui.promote_pawn(screen, row, col, current_player)
                            board.state[row][col] = current_player + choice
                        
                        current_player = 'b' if current_player == 'w' else 'w'
                        print(f"{'White' if current_player == 'w' else 'Black'}'s turn")
                            
                        selected = None
                        valid_moves = []
                    else:
                        # If clicking another piece of the same player, reselect
                        if state[row][col] != "--" and state[row][col][0] == current_player:
                            selected = (row, col)
                            valid_moves = board.get_piece_moves(row, col)
                            valid_moves = add_castling_move(board, selected, current_player, valid_moves)
                        else:
                            selected = None
                            valid_moves = []
                else:
                    if state[row][col] != "--" and state[row][col][0] == current_player:
                        selected = (row, col)
                        valid_moves = board.get_piece_moves(row, col)
                        valid_moves = add_castling_move(board, selected, current_player, valid_moves)
                    else:
                        selected = None
                        valid_moves = []

        gui.render(screen, board, current_player, selected, valid_moves, font)
        clock.tick(60)

    pg.quit()

if __name__ == '__main__':
    main()
