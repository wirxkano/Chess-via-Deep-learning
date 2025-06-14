import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run a chess game between players or agents.")
    parser.add_argument('--mode', required=True, choices=['ava', 'pva', 'pvp'],
                        help='Game mode: ava (agent vs agent), pva (player vs agent), pvp (player vs player)')
    parser.add_argument('--first', required=True, choices=['w', 'b'],
                        help='Which side goes first: w (white) or b (black)')
    parser.add_argument('--difficulty', required=True, choices=['easy', 'medium', 'hard'],
                        help='Difficulty level for AI (only used in games with agent)')

    return parser.parse_args()