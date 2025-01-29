import varseek as vk

positional_args, keyword_args = vk.utils.make_positional_arguments_list_and_keyword_arguments_dict()
vk.count(*positional_args, **keyword_args)