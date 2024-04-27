import gym
import suikaenv
from suikaenv.setting import *
env = gym.make('suika-v0')

state = env.reset()

# 環境をリセットして初期状態を取得
for i in range(10):
    print(f"episode:{i}")
    done = False
    state = env.reset()
    while not done:
        
        # ランダムなアクションを生成
        action = env.action_space.sample()
        # アクションを環境に適用し、次の状態と報酬、終了フラグを取得
        state, reward, done, _ = env.step(action)
        print(f"reward:{reward}")
        env.render()
# 環境を閉じる
env.close()