GRAVITY = 980       # 重力加速度
RESTITUTION = 0.1    # 弾性係数

WALL_ELASTICITY = 0.5 # 壁の弾性係数
WALL_FRICTION   = 1 # 壁の摩擦係数

# 色, 半径, 弾性力, 摩擦力
FRUIT_INFO = [
    ((220, 20, 60),    9, 0.2,1.0),      # サクランボ
    ((250, 128, 114), 14, 0.3,0.8),   # イチゴ
    ((186, 85, 211),  20, 0.3,0.8),    # ブドウ
    ((255, 165, 0),   25, 0.3,0.8),     # デコポン
    ((255, 140, 0),   32, 0.3,0.8),     # 柿
    ((255, 0, 0),     40, 0.3,0.8),       # リンゴ
    ((240, 230, 140), 45, 0.3,0.8),   # 梨
    ((255, 192, 203), 50, 0.3,0.8),   # 桃
    ((255, 255, 0),   60, 0.3,0.8),     # パイナップル
    ((173, 255, 47),  80, 0.3,0.8),    # メロン
    ((0, 128, 0),     95, 0.3,0.8)        # スイカ
]

BACKGROUND_COLOR = (255,255,255)

# 固定値
SCREEN_WIDTH  = 450
SCREEN_HEIGHT = 640
BOX_WIDTH  = 320
BOX_HEIGHT = 400

BOTTOM_Y = 200 + BOX_HEIGHT
SIDE_X   = [(SCREEN_WIDTH-BOX_WIDTH)//2,(SCREEN_WIDTH-BOX_WIDTH)//2 + BOX_WIDTH]
CURSOR_BOUND_MIN_X = (SCREEN_WIDTH - BOX_WIDTH) // 2 + 25
CURSOR_BOUND_MAX_X = (SCREEN_WIDTH + BOX_WIDTH) // 2 - 25


