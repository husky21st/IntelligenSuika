GRAVITY = 5.0        # 重力加速度
RESTITUTION = 0.1    # 弾性係数
ONE_SECOND_FRAME = 60 # 1秒間のフレーム数
FRUIT_COLOR_SIZE = [
    ((220, 20, 60), 9),      # サクランボ
    ((250, 128, 114), 14),   # イチゴ
    ((186, 85, 211), 20),    # ブドウ
    ((255, 165, 0), 25),     # デコポン
    ((255, 140, 0), 32),     # 柿
    ((255, 0, 0), 40),       # リンゴ
    ((240, 230, 140), 45),   # 梨
    ((255, 192, 203), 50),   # 桃
    ((255, 255, 0), 60),     # パイナップル
    ((173, 255, 47), 80),    # メロン
    ((0, 128, 0), 90)        # スイカ (サイズ：仮)
]
BOX_LINE_WIDTH = 5
BACKGROUND_COLOR = (255,255,255)

SCREEN_WIDTH  = 450
SCREEN_HEIGHT = 640
BOX_WIDTH  = 320
BOX_HEIGHT = 400

BOTTOM_Y = 200 + BOX_HEIGHT
SIDE_X   = [(SCREEN_WIDTH-BOX_WIDTH)//2,(SCREEN_WIDTH-BOX_WIDTH)//2 + BOX_WIDTH]
CURSOR_BOUND_MIN_X = (SCREEN_WIDTH - BOX_WIDTH) // 2 + 25
CURSOR_BOUND_MAX_X = (SCREEN_WIDTH + BOX_WIDTH) // 2 - 25