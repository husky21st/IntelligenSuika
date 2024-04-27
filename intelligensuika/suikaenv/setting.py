GRAVITY        = 980    # 重力加速度
RESTITUTION    = 0.1    # 弾性係数
REWARD_DEFAULT = 0.1    # デフォルトの報酬
FRAMES_PER_SECOND = 120 # 1秒間に描画するフレーム数
WAIT_FRAMES       = 120  # 観測を送るまでの待機フレーム数
PYMUNK_FPS        = 45  # pymunkの更新頻度
WALL_ELASTICITY = 0.01  # 壁の弾性係数
WALL_FRICTION   = 1     # 壁の摩擦係数
MAX_FRUIT_NUM   = 62    # 最大フルーツ数
GAME_OVER_LINE  = 240 # defalut: 240

# [色, 半径, 弾性力, 摩擦力, 報酬]
FRUIT_INFO = [
    (None), # 0番目は使わない
    ((220, 20, 60),    9, 0.3,0.85, 0.0), # サクランボ
    ((250, 128, 114), 14, 0.3,0.85, 0.1), # イチゴ
    ((186, 85, 211),  20, 0.3,0.85, 0.2), # ブドウ
    ((255, 165, 0),   25, 0.3,0.85, 0.3), # デコポン
    ((255, 140, 0),   32, 0.3,0.85, 0.4), # 柿
    ((255, 0, 0),     40, 0.3,0.85, 0.5), # リンゴ
    ((240, 230, 140), 45, 0.3,0.85, 0.6), # 梨
    ((255, 192, 203), 50, 0.3,0.85, 0.7), # 桃
    ((255, 255, 0),   60, 0.3,1.00, 0.8), # パイナップル
    ((173, 255, 47),  80, 0.3,0.90, 0.9), # メロン
    ((0, 128, 0),     95, 0.3,0.75, 1.0)  # スイカ
]


# 固定値
BACKGROUND_COLOR = (255,255,255)
SCREEN_WIDTH  = 450
SCREEN_HEIGHT = 640
BOX_WIDTH  = 320
BOX_HEIGHT = 400
BOTTOM_Y = 200 + BOX_HEIGHT
SIDE_X   = [(SCREEN_WIDTH-BOX_WIDTH)//2,(SCREEN_WIDTH-BOX_WIDTH)//2 + BOX_WIDTH]
CURSOR_BOUND_MIN_X = (SCREEN_WIDTH - BOX_WIDTH) // 2 + 25
CURSOR_BOUND_MAX_X = (SCREEN_WIDTH + BOX_WIDTH) // 2 - 25

