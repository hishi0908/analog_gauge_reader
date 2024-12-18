from ultralytics import YOLO
import numpy as np
import cv2
from scipy import odr


def segment_gauge_needle(image, model_path='best.pt'):
    """
    ファインチューニングされたYOLO v8モデルを使用して針のセグメンテーション用マスクを取得します。
    :param image: 入力画像を表すnumpy配列
    :param model_path: YOLOv8検出モデルへのパス
    :return: 針のマスクピクセルのx座標とy座標
    """
    model = YOLO(model_path)  # YOLOモデルを読み込む

    # ゲージフェイスと針を検出するために画像で推論を実行
    results = model.predict(image)

    # 検出結果から針のマスクを抽出
    try:
        # GPUが使用可能な場合、テンソルをNumPy配列に変換
        needle_mask = results[0].masks.data[0].numpy()
    except:
        # GPUが利用できない場合、CPUテンソルをNumPy配列に変換
        needle_mask = results[0].masks.data[0].cpu().numpy()

    # この部分のコードは、YOLOv8モデルの推論結果から針のマスクを取得する際に、CPUとGPUの違いを考慮しています。
    # GPUを使用する場合、テンソルはGPU上に格納されており`.numpy()`を使用してCPU上のNumPy配列に変換します。
    # 一方で、GPUが利用できない場合は`.cpu().numpy()`を使用してCPU上のテンソルをNumPy配列に変換します。
    # このような切り替えによって、コードが環境に依存せず動作する柔軟性を持っています。

    # 入力画像の寸法に一致するように針のマスクをリサイズ
    needle_mask_resized = cv2.resize(needle_mask,
                                     dsize=(image.shape[1], image.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)
    # ここでは、検出した針のマスクを元の画像と同じ寸法にリサイズしています。
    # この操作は、後続の処理が元の画像座標系で行われるようにするために必要です。
    # リサイズ方法として`INTER_NEAREST`を使用しているのは、マスクの離散値を保つためです。
  
    # 針のマスクピクセルのx座標とy座標を取得
    y_coords, x_coords = np.where(needle_mask_resized)
    # このコードでは、np.where関数を使用してリサイズされた針のマスク画像内の非ゼロ（マスクされた）ピクセルの座標を取得しています。
    # y_coords: リサイズ後のマスク画像において、非ゼロピクセルが存在する行（縦方向）のインデックスを保持します。
    # x_coords: リサイズ後のマスク画像において、非ゼロピクセルが存在する列（横方向）のインデックスを保持します。
    # この座標ペア（x_coords, y_coords）は、針のセグメンテーション結果を元に、
    # 針がどの位置にあるかを示すデータとして後続の処理（例えば線形フィッティングなど）に使用されます。

    return x_coords, y_coords


def get_fitted_line(x_coords, y_coords):
    """
    針の座標に線をフィットさせるために直交距離回帰（ODR）を実行します。
    :param x_coords: 針のマスクピクセルのx座標
    :param y_coords: 針のマスクピクセルのy座標
    :return: 線の係数と残差分散
    """
    # ODR（直交距離回帰）を利用して線形フィットを実行
    # odr.Model: 線形関数（linear）を使用したモデルの定義
    odr_model = odr.Model(linear)

    # odr.Data: フィット対象のデータとしてx座標とy座標を登録
    data = odr.Data(x_coords, y_coords)

    # odr.ODR: データとモデルを用いて回帰を実行
    # beta0: 初期値として傾き0.2、切片1.0を指定
    # maxit: 最大反復回数を600に設定
    ordinal_distance_reg = odr.ODR(data, odr_model, beta0=[0.2, 1.], maxit=600)

    # 回帰を実行し、結果を取得
    out = ordinal_distance_reg.run()

    # 線の傾き（slope）と切片（intercept）を取得
    line_coeffs = out.beta

    # 残差分散（フィットの誤差を示す指標）を取得
    residual_variance = out.res_var

    return line_coeffs, residual_variance


def linear(B, x):
    """
    ODRモデル用の線形関数を定義します。
    :param B: 線形方程式の係数 [傾き, 切片]
    :param x: x値
    :return: 線形方程式で計算されたy値
    """
    return B[0] * x + B[1]

    # 図示ように取得する。 pipeline1関数で使っている
def get_start_end_line(needle_mask):
    """
    針のマスクを特定の軸に沿って最小値と最大値を取得します。
    :param needle_mask: 針のマスク
    :return: 最小値と最大値
    """
    return np.min(needle_mask), np.max(needle_mask)

    # 図示ように取得する。 pipeline1関数で使っている
def cut_off_line(x, y_min, y_max, line_coeffs):
    """
    フィットされた線の端点を指定された境界内に調整します。
    :param x: 線の端点のx値
    :param y_min: 最小y境界
    :param y_max: 最大y境界
    :param line_coeffs: フィットされた線の係数
    :return: 調整されたx値
    """
    line = np.poly1d(line_coeffs)
    y = line(x)
    _cut_off(x, y, y_min, y_max, line_coeffs, 0)
    _cut_off(x, y, y_min, y_max, line_coeffs, 1)
    return x[0], x[1]


def _cut_off(x, y, y_min, y_max, line_coeffs, i):
    """
    境界に基づいて線の単一端点を調整するヘルパー関数。
    :param x: 線の端点のx値
    :param y: 線の端点のy値
    :param y_min: 最小y境界
    :param y_max: 最大y境界
    :param line_coeffs: フィットされた線の係数
    :param i: 調整する端点のインデックス
    """
    if y[i] > y_max:
        y[i] = y_max
        x[i] = 1 / line_coeffs[0] * (y_max - line_coeffs[1])
    if y[i] < y_min:
        y[i] = y_min
        x[i] = 1 / line_coeffs[0] * (y_min - line_coeffs[1])
