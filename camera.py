import cv2
import os
import time
import numpy as np

# フォルダ内の画像ファイルのパスを取得する関数
def get_image_paths(folder):
    image_paths = []
    for filename in os.listdir(folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_paths.append(os.path.join(folder, filename))
    return image_paths

# 予め用意した画像フォルダのパス
reference_folder = 'images/'

# 予め用意した画像フォルダ内の画像ファイルパスを取得
reference_image_paths = get_image_paths(reference_folder)

# 閾値
threshold = 0.4  # 類似度の閾値（0から1の範囲で設定）

# 関数の実行（この例では、類似度が閾値以上の場合に呼び出される関数）
def execute_function(image_name, frame):
    # カメラ画像を予め用意した画像のサイズにリサイズ
    resized_frame = cv2.resize(frame, (reference_image_width, reference_image_height))

    # 結合するためのキャンバスを作成
    canvas = np.zeros((max(reference_image_height, resized_frame.shape[0]), reference_image_width + resized_frame.shape[1], 3), dtype=np.uint8)

    # 画像をキャンバスに配置
    canvas[:reference_image_height, :reference_image_width] = cv2.imread(image_name)
    canvas[:resized_frame.shape[0], reference_image_width:] = resized_frame

    # 画像を表示
    cv2.imshow("Matched Image vs Camera Image", canvas)
    cv2.waitKey(0)

# カメラキャプチャの設定
cap = cv2.VideoCapture(2)  # 0はデフォルトのカメラデバイスを表す
reference_image = cv2.imread(reference_image_paths[0])
reference_image_height, reference_image_width, _ = reference_image.shape

while True:
    # カメラからフレームをキャプチャ
    ret, frame = cap.read()
    # フレームをトリミング
    cropped_frame = frame[216:960, 1080:1980]  # 例として、(100, 100)から(300, 300)までをトリミング

    gray_cropped_frame = cropped_frame

    # トリミングされた画像の平均ピクセル値を計算
    mean_pixel_value = np.mean(gray_cropped_frame)

    # 真っ黒の画像の場合は類似度計算をスキップ
    if mean_pixel_value == 0:
        continue

    # トリミングされた画像と予め用意した画像の類似度を計算
    for reference_image_path in reference_image_paths:
        reference_image = cv2.imread(reference_image_path)
        similarity_score = cv2.matchTemplate(gray_cropped_frame, reference_image, cv2.TM_CCOEFF_NORMED)
        max_similarity = np.max(similarity_score)
        print(max_similarity)
        if max_similarity >= threshold and max_similarity != 1:
            cv2.imshow('Cropped Frame', reference_image)
            image_name = os.path.basename(reference_image_path)
            execute_function(reference_image_path, frame)

    # トリミングされた画像を表示
    cv2.imshow('Cropped Frame', cropped_frame)

    # 'q'を押したら終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# キャプチャを解放し、ウィンドウを閉じる
cap.release()
cv2.destroyAllWindows()
