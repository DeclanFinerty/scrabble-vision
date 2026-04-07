import cv2
import numpy as np

board = cv2.imread("data/eval/eval_board.jpeg")
grid = cv2.imread("data/eval/debug/eval_board_grid.jpg")

# Resize both to same height
h = 800
board = cv2.resize(board, (int(board.shape[1] * h / board.shape[0]), h))
grid = cv2.resize(grid, (int(grid.shape[1] * h / grid.shape[0]), h))

# Add gap and combine
gap = np.ones((h, 20, 3), dtype=np.uint8) * 255
combined = np.hstack([board, gap, grid])

cv2.imwrite("assets/eval_result.jpg", combined)