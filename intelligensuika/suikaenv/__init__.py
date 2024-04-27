from gymnasium.envs.registration import register
register(
    id='suika-v0',
    entry_point='suikaenv.gym_game_field:SuikaEnv'
)