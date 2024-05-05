import math
import time
import numpy
import pandas
import pygame
from math import inf as infinity


class Node:
    def __init__(self, init_coordinates, final_coordinates, board):
        self.init_coordinates = init_coordinates
        self.final_coordinates = final_coordinates
        self.board = board
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def __repr__(self):
        return f"Node(init_coordinates={self.init_coordinates}, final_coords={self.final_coordinates}, board={self.board})"


def init_board():
    w, h = 16, 16
    board = [[0 for x in range(w)] for y in range(h)]

    # player 1 fields
    starting_coordinates_one = [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [2, 0],
                                [2, 1], [2, 2], [2, 3], [3, 0], [3, 1], [3, 2], [4, 0], [4, 1]]
    for coordinate_set in starting_coordinates_one:
        board[coordinate_set[0]][coordinate_set[1]] = 1

    # player 2 fields
    starting_coordinates_two = [[11, 14], [11, 15], [12, 13], [12, 14], [12, 15], [13, 12], [13, 13], [13, 14],
                                [13, 15], [14, 11], [14, 12], [14, 13], [14, 14], [14, 15], [15, 11], [15, 12],
                                [15, 13], [15, 14], [15, 15]]
    for coordinate_set in starting_coordinates_two:
        board[coordinate_set[0]][coordinate_set[1]] = 2

    return board


def display_board(board):
    x = numpy.array(board)
    labels = range(16)
    df = pandas.DataFrame(x, columns=labels, index=labels)
    print(df)


def get_possible_moves(board, coordinate_x, coordinate_y):
    possible_moves = list()
    options = [[coordinate_x - 1, coordinate_y - 1], [coordinate_x - 1, coordinate_y],
               [coordinate_x - 1, coordinate_y + 1],
               [coordinate_x, coordinate_y - 1], [coordinate_x, coordinate_y + 1],
               [coordinate_x + 1, coordinate_y - 1], [coordinate_x + 1, coordinate_y],
               [coordinate_x + 1, coordinate_y + 1]]

    for option in options:
        if option[0] in range(16) and option[1] in range(16):
            if board[option[0]][option[1]] == 0:
                possible_moves.append(option)
            else:
                if coordinate_x == option[0] and coordinate_y < option[1] and (option[1] + 1) in range(16) and \
                        board[option[0]][option[1] + 1] == 0:
                    possible_moves.append([option[0], option[1] + 1])
                elif coordinate_x == option[0] and coordinate_y > option[1] and (option[1] - 1) in range(16) and \
                        board[option[0]][option[1] - 1] == 0:
                    possible_moves.append([option[0], option[1] - 1])
                elif coordinate_x < option[0] and coordinate_y > option[1] and (option[0] + 1) in range(16) and \
                        (option[1] - 1) in range(16) and board[option[0] + 1][option[1] - 1] == 0:
                    possible_moves.append([option[0] + 1, option[1] - 1])
                elif coordinate_x < option[0] and option[1] == coordinate_y and (option[0] + 1) in range(16) and \
                        board[option[0] + 1][option[1]] == 0:
                    possible_moves.append([option[0] + 1, option[1]])
                elif coordinate_x < option[0] and coordinate_y < option[1] and (option[0] + 1) in range(16) and (
                        option[1] + 1) in range(16) and board[option[0] + 1][option[1] + 1] == 0:
                    possible_moves.append([option[0] + 1, option[1] + 1])
                elif coordinate_x > option[0] and coordinate_y > option[1] and (option[0] - 1) in range(16) and \
                        (option[1] - 1) in range(16) and board[option[0] - 1][option[1] - 1] == 0:
                    possible_moves.append([option[0] - 1, option[1] - 1])
                elif coordinate_x > option[0] and coordinate_y == option[1] and (option[0] - 1) in range(16) and \
                        board[option[0] - 1][option[1]] == 0:
                    possible_moves.append([option[0] - 1, option[1]])
                elif coordinate_x > option[0] and coordinate_y < option[1] and (option[0] - 1) in range(16) and (
                        option[1] + 1) in range(16) and board[option[0] - 1][option[1] + 1] == 0:
                    possible_moves.append([option[0] - 1, option[1] + 1])
    return possible_moves


def get_possible_jumps(board, coordinate_x, coordinate_y, jumped_coordinates):
    possible_moves = list()
    options = [[coordinate_x - 1, coordinate_y - 1], [coordinate_x - 1, coordinate_y],
               [coordinate_x - 1, coordinate_y + 1],
               [coordinate_x, coordinate_y - 1], [coordinate_x, coordinate_y + 1],
               [coordinate_x + 1, coordinate_y - 1], [coordinate_x + 1, coordinate_y],
               [coordinate_x + 1, coordinate_y + 1]]
    for option in options:
        if option[0] in range(16) and option[1] in range(16):
            if board[option[0]][option[1]] != 0 and option not in jumped_coordinates:
                if coordinate_x == option[0] and coordinate_y < option[1] and (option[1] + 1) in range(16) and \
                        board[option[0]][option[1] + 1] == 0:
                    possible_moves.append([option[0], option[1] + 1])
                elif coordinate_x == option[0] and coordinate_y > option[1] and (option[1] - 1) in range(16) and \
                        board[option[0]][option[1] - 1] == 0:
                    possible_moves.append([option[0], option[1] - 1])
                elif coordinate_x < option[0] and coordinate_y > option[1] and (option[0] + 1) in range(16) and \
                        (option[1] - 1) in range(16) and board[option[0] + 1][option[1] - 1] == 0:
                    possible_moves.append([option[0] + 1, option[1] - 1])
                elif coordinate_x < option[0] and option[1] == coordinate_y and (option[0] + 1) in range(16) and \
                        board[option[0] + 1][option[1]] == 0:
                    possible_moves.append([option[0] + 1, option[1]])
                elif coordinate_x < option[0] and coordinate_y < option[1] and (option[0] + 1) in range(16) and (
                        option[1] + 1) in range(16) and board[option[0] + 1][option[1] + 1] == 0:
                    possible_moves.append([option[0] + 1, option[1] + 1])
                elif coordinate_x > option[0] and coordinate_y > option[1] and (option[0] - 1) in range(16) and \
                        (option[1] - 1) in range(16) and board[option[0] - 1][option[1] - 1] == 0:
                    possible_moves.append([option[0] - 1, option[1] - 1])
                elif coordinate_x > option[0] and coordinate_y == option[1] and (option[0] - 1) in range(16) and \
                        board[option[0] - 1][option[1]] == 0:
                    possible_moves.append([option[0] - 1, option[1]])
                elif coordinate_x > option[0] and coordinate_y < option[1] and (option[0] - 1) in range(16) and (
                        option[1] + 1) in range(16) and board[option[0] - 1][option[1] + 1] == 0:
                    possible_moves.append([option[0] - 1, option[1] + 1])
    return possible_moves


def get_jumped_coordinates(prev_h, prev_w, h, w):
    if prev_h - h < 0:
        jumped_h_coordinate = prev_h + 1
    elif prev_h - h > 0:
        jumped_h_coordinate = prev_h - 1
    else:
        jumped_h_coordinate = prev_h

    if prev_w - w < 0:
        jumped_w_coordinate = prev_w + 1
    elif prev_w - w > 0:
        jumped_w_coordinate = prev_w - 1
    else:
        jumped_w_coordinate = prev_w

    return [jumped_h_coordinate, jumped_w_coordinate]


def check_game_end(board, player_one):
    winning_coordinates_two = [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [2, 0],
                               [2, 1], [2, 2], [2, 3], [3, 0], [3, 1], [3, 2], [4, 0], [4, 1]]
    winning_coordinates_one = [[11, 14], [11, 15], [12, 13], [12, 14], [12, 15], [13, 12], [13, 13], [13, 14],
                               [13, 15], [14, 11], [14, 12], [14, 13], [14, 14], [14, 15], [15, 11], [15, 12],
                               [15, 13], [15, 14], [15, 15]]
    if player_one:
        player_one_win = True
        for coordinates in winning_coordinates_one:
            if board[coordinates[0]][coordinates[1]] != 1:
                player_one_win = False
                return player_one_win
        return player_one_win
    elif not player_one:
        player_two_win = True
        for coordinates in winning_coordinates_two:
            if board[coordinates[0]][coordinates[1]] != 2:
                player_two_win = False
                return player_two_win
        return player_two_win


def evaluate_by_distance(board):
    """calculate a distance to nearest finish field for AI"""
    return 1


def get_all_moves(board, player):
    for x in range(16):
        for y in range(16):
            if board[x][y] == player:
                # found AI pawn
                # x = h, y = w
                print(board[x][y])
                print(x, y)
                possible_moves = get_possible_moves_ai(board, x, y, player)
                print(possible_moves)
                return possible_moves


def min_max_algorithm(board, depth, player):
    """MinMax algorithm - takes a board algorithm with current board state
    and returns the best possible move for a player according to a cost function
    board - current board state
    depth - limiting number of moves to check
    player - 1 or 2 ( MAX or MIN )"""
    print(board)
    # moves_tree = list()
    root = Node(None, None, board)
    while depth:
        for x in range(16):
            for y in range(16):
                if board[x][y] == player:
                    # found AI pawn
                    # x = h, y = w
                    print(board[x][y])
                    print(x, y)
                    possible_moves = get_possible_moves_ai(board, x, y, player)
                    print(possible_moves)
                    # moves_tree.append([[x, y], possible_moves])
                    for move in possible_moves:
                        new_board = [row[:] for row in board]  # Create a copy of the board
                        new_board[x][y] = 0  # Remove the piece from its original position
                        new_board[move[0]][
                            move[1]] = player  # Move the piece to the new position (2 cuz ai player is 2)
                        move1 = Node([x, y], move, new_board)
                        root.add_child(move1)

    if player == 2:
        player = 1
    else:
        player = 2
    depth -= 1

    # print(moves_tree)
    # Traverse the tree starting from the root
    # dfs(root)


def min_max_algorithm_v2(board, depth, player, starting_depth):
    if depth == 0:
        # return evaluate(board)  # Assume evaluate is a function you will define to score the board.
        print("over")
        return 1

    best_move = None
    best_board = None
    if player == 2:
        max_evaluation = float('-inf')
        for x in range(16):
            for y in range(16):
                if board[x][y] == player:
                    moves = get_possible_moves_ai(board, x, y, player)
                    for move in moves:
                        new_board = [row[:] for row in board]
                        new_board[x][y] = 0
                        new_board[move[0]][move[1]] = player
                        evaluation = min_max_algorithm_v2(new_board, depth - 1, 1, starting_depth)
                        if evaluation > max_evaluation:
                            max_evaluation = evaluation
                            best_move = move
                            best_board = new_board
        return max_evaluation if depth != starting_depth else best_board
    else:
        min_evaluation = float('inf')
        for x in range(16):
            for y in range(16):
                if board[x][y] == player:
                    moves = get_possible_moves_ai(board, x, y, player)
                    for move in moves:
                        new_board = [row[:] for row in board]
                        new_board[x][y] = 0
                        new_board[move[0]][move[1]] = player
                        evaluation = min_max_algorithm_v2(new_board, depth - 1, 2, starting_depth)
                        if evaluation < min_evaluation:
                            min_evaluation = evaluation
                            best_move = move
                            best_board = new_board
        return min_evaluation if depth != starting_depth else best_board


def dfs(node):
    print(node)  # Process the node
    for child in node.children:
        dfs(child)  # Recursively visit children


def get_possible_moves_ai(board, coordinate_x, coordinate_y, player, jumped_coordinates=[]):
    possible_jumps = []

    if not jumped_coordinates:  # First move, consider only regular moves
        possible_moves = get_possible_moves(board, coordinate_x, coordinate_y)
        for move in possible_moves:
            new_x, new_y = move
            new_board = [row[:] for row in board]  # Create a copy of the board
            new_board[coordinate_x][coordinate_y] = 0  # Remove the piece from its original position
            new_board[new_x][new_y] = player  # Move the piece to the new position (2 cuz ai player is 2)
            further_jumps = get_possible_moves_ai(new_board, new_x, new_y, player,
                                                  jumped_coordinates + [[new_x, new_y]])
            if further_jumps:
                possible_jumps.extend(further_jumps)
            else:
                possible_jumps.append([new_x, new_y])
    else:  # Subsequent moves, consider only jump moves
        for option in [[coordinate_x - 2, coordinate_y - 2], [coordinate_x - 2, coordinate_y],
                       [coordinate_x - 2, coordinate_y + 2], [coordinate_x, coordinate_y - 2],
                       [coordinate_x, coordinate_y + 2], [coordinate_x + 2, coordinate_y - 2],
                       [coordinate_x + 2, coordinate_y], [coordinate_x + 2, coordinate_y + 2]]:
            if option[0] in range(16) and option[1] in range(16) and option not in jumped_coordinates:
                new_x, new_y = option
                jumped_x = (new_x + coordinate_x) // 2
                jumped_y = (new_y + coordinate_y) // 2
                if 0 <= jumped_x < 16 and 0 <= jumped_y < 16 and board[jumped_x][jumped_y] != 0 and board[new_x][
                    new_y] == 0:
                    new_board = [row[:] for row in board]  # Create a copy of the board
                    new_board[coordinate_x][coordinate_y] = 0  # Remove the piece from its original position
                    new_board[new_x][new_y] = player  # Move the piece to the new position (2 cuz ai player is 2)
                    further_jumps = get_possible_moves_ai(new_board, new_x, new_y, player,
                                                          jumped_coordinates + [[new_x, new_y]])
                    if further_jumps:
                        possible_jumps.extend(further_jumps)
                    else:
                        possible_jumps.append([new_x, new_y])

    return possible_jumps


def game(board_with_data):
    pygame.init()
    screen = pygame.display.set_mode((720, 720))
    pygame.display.set_caption("Halma")
    clock = pygame.time.Clock()
    board = pygame.Surface((720, 720))
    moves = pygame.Surface((720, 720))
    moves.set_colorkey((0, 0, 0))
    over_font = pygame.font.Font('freesansbold.ttf', 82)
    over_text = over_font.render("GAME OVER", True, (0, 0, 0))

    # draw the board
    def board_update():
        color_swap = True
        finish_fields = [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [2, 0],
                         [2, 1], [2, 2], [2, 3], [3, 0], [3, 1], [3, 2], [4, 0], [4, 1], [11, 14], [11, 15], [12, 13],
                         [12, 14], [12, 15], [13, 12], [13, 13], [13, 14],
                         [13, 15], [14, 11], [14, 12], [14, 13], [14, 14], [14, 15], [15, 11], [15, 12],
                         [15, 13], [15, 14], [15, 15]]
        for x in range(16):
            for y in range(16):
                if color_swap:
                    if [x, y] in finish_fields:
                        pygame.draw.rect(board, (84, 72, 56), (x * 45, y * 45, 45, 45))
                    else:
                        pygame.draw.rect(board, (210, 180, 140), (x * 45, y * 45, 45, 45))
                else:
                    if [x, y] in finish_fields:
                        pygame.draw.rect(board, (179, 144, 111), (x * 45, y * 45, 45, 45))
                    else:
                        pygame.draw.rect(board, (255, 206, 158), (x * 45, y * 45, 45, 45))
                color_swap = not color_swap
            color_swap = not color_swap

        for x in range(16):
            for y in range(16):
                if board_with_data[y][x] == 1:
                    pygame.draw.circle(board, (0, 0, 0), ((x + 1) * 45 - 45 / 2, (y + 1) * 45 - 45 / 2), 20)
                if board_with_data[y][x] == 2:
                    pygame.draw.circle(board, (255, 255, 255), ((x + 1) * 45 - 45 / 2, (y + 1) * 45 - 45 / 2), 20)

    def update_possible_moves(possible_moves=None):
        moves.fill((0, 0, 0))
        if possible_moves is not None:
            for move in possible_moves:
                pygame.draw.rect(moves, (125, 125, 125), (move[1] * 45 + 5, move[0] * 45 + 5, 35, 35))

    board_update()
    running = True
    turn_player_one = True
    possible_move_coordinates = None
    jumped_coordinates = list()
    prev_w = None
    prev_h = None
    continue_jumps = False
    game_over = False
    while running:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN and not game_over and turn_player_one:
                pos = pygame.mouse.get_pos()
                if pos[0] in range(0, 720) and pos[1] in range(0, 720):
                    w = math.floor(pos[0] / 45)
                    h = math.floor(pos[1] / 45)
                    print(w, h)
                    if board_with_data[h][w] == 1 and turn_player_one and not continue_jumps:
                        # handles move selection for player one before jumps are forced
                        possible_move_coordinates = get_possible_moves(board_with_data, h, w)
                        print(possible_move_coordinates)
                        update_possible_moves(possible_moves=possible_move_coordinates)
                        prev_w = w
                        prev_h = h
                    elif possible_move_coordinates is not None and [h,
                                                                    w] in possible_move_coordinates and turn_player_one:
                        # handles move confirmation for player one and possibly forces jumps
                        print("move")
                        print("prev_h = ", prev_h)
                        print("prev_w = ", prev_w)
                        board_with_data[prev_h][prev_w] = 0
                        if turn_player_one:
                            board_with_data[h][w] = 1
                        else:
                            board_with_data[h][w] = 2

                        print(prev_h - h)
                        print(prev_w - w)

                        if abs(prev_h - h) == 2 or abs(prev_w - w == 2):
                            jumped_coordinates.append(get_jumped_coordinates(prev_h, prev_w, h, w))
                            possible_move_coordinates = get_possible_jumps(board_with_data, h, w, jumped_coordinates)
                            if len(possible_move_coordinates) != 0:
                                continue_jumps = True
                                update_possible_moves(possible_moves=possible_move_coordinates)
                                board_update()
                                prev_w = w
                                prev_h = h
                            else:
                                continue_jumps = False
                                jumped_coordinates = list()
                                prev_w = None
                                prev_h = None
                                if check_game_end(board_with_data, turn_player_one):
                                    game_over = True
                                    break
                                turn_player_one = not turn_player_one
                                possible_move_coordinates = None
                                board_update()
                                update_possible_moves()
                        else:
                            continue_jumps = False
                            jumped_coordinates = list()
                            prev_w = None
                            prev_h = None
                            if check_game_end(board_with_data, turn_player_one):
                                game_over = True
                                break
                            turn_player_one = not turn_player_one
                            possible_move_coordinates = None
                            board_update()
                            update_possible_moves()
            elif not turn_player_one:
                # handles AI turn
                # step 1 : call minmax algorithm - analyze all possible moves for each pawns
                #  get current board status
                board_with_data = min_max_algorithm_v2(board_with_data, 3, 2, 3)
                board_update()
                turn_player_one = not turn_player_one

        screen.fill((255, 255, 255))
        screen.blit(board, (0, 0))
        screen.blit(moves, (0, 0))
        if game_over:
            screen.blit(over_text, (100, 300))
        pygame.display.update()


def run():
    start_time = time.time()
    board = init_board()
    game(board)
    end_time = time.time()
    print("Total time: ", end_time - start_time)


if __name__ == '__main__':
    run()
