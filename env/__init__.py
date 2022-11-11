from gym.envs.registration import register
from env.lunar_lander import LunarLanderEnv
from env.pendulum import PendulumEnv

register(
    id="LunarLander-v3",
    entry_point="env:LunarLanderEnv",
)

"""
    Registered environments
"""
register(
    id="Pendulum-v2",
    #entry_point="%s:CustomHopper" % __name__,
    entry_point="env:PendulumEnv",
    #max_episode_steps=500,
)

register(
    id="Pendulum-source-v2",
    entry_point="env:PendulumEnv",
    #max_episode_steps=500,
    kwargs={"domain": "source"}
)

register(
    id="Pendulum-target-v2",
    entry_point="env:PendulumEnv",
    #max_episode_steps=500,
    kwargs={"domain": "target"}
)