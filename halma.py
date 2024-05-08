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


def min_max_algorithm(board, depth, player, starting_depth, alpha=float('-inf'), beta=float('inf'), path=None):
    if path is None:
        path = []  # Initialize path at the top call

    if depth == 0:
        # return evaluate(board)  # Assume evaluate is a function you will define to score the board.
        value = rapid_strategy_evaluate(board, player)
        return value, path

    best_board = None
    best_path = []
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
                        new_path = path + [[x, y], [move[0], move[1]]]
                        evaluation, current_path = min_max_algorithm(new_board, depth - 1, 1, starting_depth, alpha,
                                                                     beta, new_path)
                        if evaluation > max_evaluation:
                            max_evaluation = evaluation
                            best_board = new_board
                            best_path = current_path
                        alpha = max(alpha, evaluation)
                        if beta <= alpha:
                            break
        return (max_evaluation, best_path) if depth != starting_depth else (best_board, best_path)
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
                        new_path = path + [[x, y], [move[0], move[1]]]
                        evaluation, current_path = min_max_algorithm(new_board, depth - 1, 2, starting_depth, alpha,
                                                                     beta, new_path)
                        if evaluation < min_evaluation:
                            min_evaluation = evaluation
                            best_board = new_board
                            best_path = current_path
                        beta = min(beta, evaluation)
                        if beta <= alpha:
                            break
        return (min_evaluation, best_path) if depth != starting_depth else (best_board, best_path)


def rapid_strategy_evaluate(board, opponent):
    player = 3 - opponent
    goal_coordinates = [0, 0] if player == 2 else [15, 15]
    total_distance = 0

    for x in range(16):
        for y in range(16):
            if board[x][y] == player:
                distance = manhattan_distance(x, y, goal_coordinates[0], goal_coordinates[1])
                total_distance += distance

    # using 570 as the distance value should never be higher than that
    return 570 - total_distance


def congestion_strategy_evaluate(board, opponent):
    # number of pieces clustered implies better movement available (more jumps)
    player = 3 - opponent
    adjacent_count = 0
    directions = [
        (1, 0), (-1, 0), (0, 1), (0, -1),  # Horizontal and vertical
        (1, 1), (1, -1), (-1, 1), (-1, -1)  # Diagonal
    ]
    for x in range(16):
        for y in range(16):
            if board[x][y] == player:
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < 16 and 0 <= ny < 16 and board[nx][ny] != 0:
                        adjacent_count += 1
    return adjacent_count


def center_strategy_evaluate(board, opponent):
    player = 3 - opponent
    goal_coordinates = [0, 0] if player == 2 else [15, 15]

    total_distance_eval = rapid_strategy_evaluate(board, opponent)
    center_score = center_control(board, player, goal_coordinates)

    # Combine distance and center scores
    score = total_distance_eval * 1.5 + center_score * 0.5

    return score


def center_control(board, player, goal):
    center_score = 0
    max_distance = max(goal[0], goal[1])
    center_x, center_y = len(board) // 2, len(board[0]) // 2
    range_x = len(board) // 4
    range_y = len(board[0]) // 4

    for x in range(center_x - range_x, center_x + range_x):
        for y in range(center_y - range_y, center_y + range_y):
            if board[x][y] == player:
                # Decrease score contribution as pieces move closer to goal
                distance_to_goal = abs(goal[0] - x) + abs(goal[1] - y)
                center_score += max_distance - distance_to_goal

    return center_score


def manhattan_distance(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)


def get_possible_moves_ai(board, coordinate_x, coordinate_y, player, jumped_coordinates=[]):
    possible_jumps = []

    if not jumped_coordinates:  # First move, consider only regular moves
        jumped_coordinates = jumped_coordinates + [[coordinate_x, coordinate_y]]
        possible_moves = get_possible_moves(board, coordinate_x, coordinate_y)
        for move in possible_moves:
            new_x, new_y = move
            if abs(coordinate_x - new_x) == 2 or abs(coordinate_y - new_y == 2):
                new_board = [row[:] for row in board]  # Create a copy of the board
                new_board[coordinate_x][coordinate_y] = 0  # Remove the piece from its original position
                new_board[new_x][new_y] = player  # Move the piece to the new position (2 cuz ai player is 2)
                further_jumps = get_possible_moves_ai(new_board, new_x, new_y, player,
                                                      jumped_coordinates + [[new_x, new_y]])
                if further_jumps:
                    possible_jumps.extend(further_jumps)
                else:
                    possible_jumps.append([new_x, new_y])
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
    past_move = None
    final_position = None

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

        if past_move:
            pygame.draw.rect(board, (172, 83, 83), (past_move[1] * 45, past_move[0] * 45, 45, 45))
        if final_position:
            pygame.draw.rect(board, (153, 102, 102), (final_position[1] * 45, final_position[0] * 45, 45, 45))

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
                        past_move = None
                        final_position = None
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

                        if abs(prev_h - h) == 2 or abs(prev_w - w) == 2:
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
                # board_with_data_old = board_with_data
                board_with_data, path = min_max_algorithm(board_with_data, 3, 2, 3)
                print(path)
                past_move = path[0]
                final_position = path[1]
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
