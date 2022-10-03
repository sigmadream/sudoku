import cv2


class SudokuSolver:
    def __init__(self, board):
        self.board = board

    def print_board(self):
        for i in range(9):
            for j in range(9):
                print(self.board[i][j], end=" ")
            print("\n")

    def empty(self):
        for i in range(9):
            for j in range(9):
                if self.board[i][j] == 0:
                    return i, j
        return None

    def is_valid(self, num, row, col):
        for i in range(9):
            if self.board[row][i] == num and i != col:
                return False
            if self.board[i][col] == num and i != row:
                return False
        y = row // 3
        x = col // 3
        for i in range(y * 3, y * 3 + 3):
            for j in range(x * 3, x * 3 + 3):
                if self.board[i][j] == num and (i != row or j != col):
                    return False
        return True

    def solve(self):
        pos = self.empty()
        if pos == None:
            return True
        else:
            row, col = pos
        for i in range(1, 10):
            self.board[row][col] = i
            if self.is_valid(i, row, col) and self.solve():
                return True
            else:
                self.board[row][col] = 0
        return False


def display_solution(image, final, predictions):
    image = image.copy()
    cell_width = image.shape[1] // 9
    cell_height = image.shape[0] // 9
    counter = 0
    for i in range(9):
        for j in range(9):
            if predictions[counter] != 0:
                color = (0, 0, 0)
            else:
                color = (255, 0, 0)
            if final[i][j] == 0:
                print("Couldn't properly solve!")
                return None

            text = str(final[i][j])
            offset_x = cell_width // 15
            offset_y = cell_height // 15
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_height, text_width), baseline = cv2.getTextSize(
                text, font, fontScale=1, thickness=3
            )
            bottom_left = cell_width * j + (cell_width - text_width) // 2 + offset_x
            bottom_right = (
                    cell_height * (i + 1) - (cell_height - text_height) // 2 + offset_y
            )
            image = cv2.putText(
                image,
                text,
                (int(bottom_left), int(bottom_right)),
                font,
                1,
                color,
                thickness=3,
                lineType=cv2.LINE_AA,
            )
            counter += 1
    return image


def unwarp(image, coords):
    ratio = 1.0
    tl, tr, br, bl = coords
    width_a = np.sqrt((tl[1] - tr[1]) ** 2 + (tl[0] - tr[1]) ** 2)
    width_b = np.sqrt((bl[1] - br[1]) ** 2 + (bl[0] - br[1]) ** 2)
    width = max(width_a, width_b) * ratio
    height = width

    destination = np.array([
        [0, 0],
        [height, 0],
        [height, width],
        [0, width]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(coords, destination)
    unwarped = cv2.warpPerspective(image, M, (int(height), int(width)), flags=cv2.WARP_INVERSE_MAP)
    return unwarped


def solve_sudoku(predictions, coords):
    board = np.array(predictions).reshape((9, 9))
    print(board)
    print("Solving...")
    solver = SudokuSolver(board)
    solver.solve()
    final = solver.board
    if 0 in final:
        print("Error occured while solving, try another image!")
    else:
        print(final)
    solution_board = cv2.imread("./blank.png")
    solution_image = display_solution(solution_board, final, predictions)
    return solution_image
