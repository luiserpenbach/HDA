import json
from data_lib import config_loader

config = config_loader.load_config("IGN_C1_HotFire")
col_ch_config = config.get("channel_config").get("")
print(col_ch_config.get)