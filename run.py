from preprocessing import *
from NumberRecognition import *
from SudokuSolver import *

if __name__ == "__main__":
    # 1. 변수 선언
    images, predictions = [], []
    # 2. 이미지 파일 불러오기
    filepath = input("Enter an image filepath : ")
    image = cv2.imread(filepath)
    # 3. 이미지 전처리
    preprocess = cv2.resize(preprocess_image(image), (540, 540))
    preprocess = cv2.bitwise_not(preprocess, preprocess)
    coords = get_coords(preprocess)
    preprocess = cv2.cvtColor(preprocess, cv2.COLOR_GRAY2BGR)
    coords_image = preprocess.copy()

    for coord in coords:
        cv2.circle(coords_image, (int(coord[0]), int(coord[1])), 5, (255, 0, 0), -1)

    warpedImage = warp(preprocess, coords)
    rects = display_grid(warpedImage)
    tiles = extract_grid(warpedImage, rects)

    for i, tile in enumerate(tiles):
        preprocess = preprocess_image(tile)
        flag, centered = centering_image(preprocess)
        centered_image = cv2.resize(centered, (32, 32))
        images.append(centered_image)
        # 4. 스도쿠 이미지의 숫자 인식
        centered_image = torch.Tensor(centered_image).unsqueeze(dim=0).unsqueeze(dim=0)

        preds = model(centered_image)
        _, prediction = torch.max(preds, dim=1)
        if flag:
            predictions.append(prediction.item() + 1)
        else:
            predictions.append(0)

    board = np.array(predictions).reshape((9, 9))
    print(board)
    # 5. 스도쿠 문제 풀이
    print("Solving...")
    solver = SudokuSolver(board)
    solver.solve()
    final = solver.board
    if 0 in final:
        print("Error occurred while solving, try another image!")
    else:
        # 6. 스도쿠 문제 해결
        print(final)
        solutionBoard = cv2.imread("./boards/blank.png")
        solutionImage = display_solution(solutionBoard, final, predictions)
        print("Press 'q' to quit...")
        while True:
            # cv2.imshow("Original Image", image)
            # cv2.imshow("Warped Image", warpedImage)
            # cv2.imshow("Coordinates", coordsImage)
            cv2.imshow("Solution", solutionImage)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break
