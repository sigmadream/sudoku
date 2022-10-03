import cv2
import numpy as np


def display_grid(image):
    cell_height = image.shape[0] // 9
    cell_width = image.shape[1] // 9
    indentation = 0
    rects = []

    for i in range(9):
        for j in range(9):
            p1 = (j * cell_height + indentation, i * cell_width + indentation)
            p2 = (
                (j + 1) * cell_height - indentation,
                (i + 1) * cell_width - indentation,
            )
            rects.append((p1, p2))
            cv2.rectangle(image, p1, p2, (0, 255, 0), 3)
    return rects


def get_contours(image):
    image = image.copy()
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    polygon = all_contours[0]
    return contours, polygon


def get_coords(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    polygon = all_contours[0]
    sums = []
    diffs = []

    for point in polygon:
        for x, y in point:
            sums.append(x + y)
            diffs.append(x - y)

    top_left = polygon[np.argmin(sums)].squeeze()
    bottom_right = polygon[np.argmax(sums)].squeeze()
    top_right = polygon[np.argmax(diffs)].squeeze()
    bottom_left = polygon[np.argmin(diffs)].squeeze()

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def warp(image, coords):
    ratio = 1.2
    tl, tr, br, bl = coords
    width_a = np.sqrt((tl[1] - tr[1]) ** 2 + (tl[0] - tr[1]) ** 2)
    width_b = np.sqrt((bl[1] - br[1]) ** 2 + (bl[0] - br[1]) ** 2)
    width = max(width_a, width_b) * ratio
    height = width

    destination = np.array(
        [[0, 0], [height, 0], [height, width], [0, width]], dtype=np.float32
    )
    M = cv2.getPerspectiveTransform(coords, destination)
    warped = cv2.warpPerspective(image, M, (int(height), int(width)))
    return warped


def extract_grid(image, rects):
    tiles = []
    for coords in rects:
        rect = image[coords[0][1]: coords[1][1], coords[0][0]: coords[1][0]]
        tiles.append(rect)
    return tiles
