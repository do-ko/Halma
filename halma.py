import math
import time
import numpy
import pandas
import pygame

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
            if event.type == pygame.MOUSEBUTTONDOWN and not game_over:
                pos = pygame.mouse.get_pos()
                if pos[0] in range(0, 720) and pos[1] in range(0, 720):
                    w = math.floor(pos[0] / 45)
                    h = math.floor(pos[1] / 45)
                    print(w, h)
                    if board_with_data[h][w] == 1 and turn_player_one and not continue_jumps:
                        possible_move_coordinates = get_possible_moves(board_with_data, h, w)
                        print(possible_move_coordinates)
                        update_possible_moves(possible_moves=possible_move_coordinates)
                        prev_w = w
                        prev_h = h
                    elif board_with_data[h][w] == 2 and not turn_player_one and not continue_jumps:
                        possible_move_coordinates = get_possible_moves(board_with_data, h, w)
                        print(possible_move_coordinates)
                        update_possible_moves(possible_moves=possible_move_coordinates)
                        prev_w = w
                        prev_h = h
                    elif possible_move_coordinates is not None and [h, w] in possible_move_coordinates:
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
