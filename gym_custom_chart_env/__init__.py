from gymnasium.envs.registration import register

register(
    id='gym_custom_chart_env/ChartEnv-v0',
    entry_point='gym_custom_chart_env.envs:ChartEnv',
)