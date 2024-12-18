import varseek as vk

config_file_path = "/home/jrich/Desktop/varseek/test_config.json"
params = vk.utils.load_params(config_file_path)

vk.ref(**params)