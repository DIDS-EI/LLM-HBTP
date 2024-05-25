
env_map = {}

from btgym.envs.virtualhome.envs.watch_tv_env import WatchTVEnv as VHWatchTVEnv
from btgym.envs.virtualhome.envs.milk_fridge_env import MilkFridgeEnv as MilkFridgeEnv
from btgym.envs.virtualhome.envs.test_env import TestEnv as TestEnv

from btgym.envs.virtualhometext.envs.watch_tv_env import WatchTVEnv as VHTWatchTVEnv
from btgym.envs.virtualhometext.envs.milk_frige_env import MilkFridgeEnv as VHTMilkFridgeEnv
from btgym.envs.virtualhometextsmall.envs.small_env import SmallEnv as SmallEnv
from btgym.envs.robowaiter.envs.rw_env import RWEnv as RWEnv


vh_env_map = {
    "VH-WatchTV": VHWatchTVEnv,

    "VH-PutMilkInFridge":MilkFridgeEnv,
    "VH-Test": TestEnv,

    "VHT-WatchTV": VHTWatchTVEnv,
    "VHT-PutMilkInFridge": VHTMilkFridgeEnv,
    "VHT-Small": SmallEnv,

    "RWEnv":RWEnv
}

env_map.update(vh_env_map)