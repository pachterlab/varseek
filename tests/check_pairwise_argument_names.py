import inspect
import varseek as vk
from varseek.varseek_ref import varseek_ref_only_allowable_kb_ref_arguments
from varseek.varseek_count import varseek_ref_only_allowable_kb_count_arguments
from itertools import combinations

def print_all_shared_arguments_between_pairs_of_functions(function_name_and_key_list_of_tuples, varseek_ref_only_allowable_kb_arguments, kb_name):
    function_parameters_dict_of_sets = {}
    for function_name, function_key in function_name_and_key_list_of_tuples:
        function_parameters = set(inspect.signature(function_name).parameters.keys())
        function_parameters_dict_of_sets[function_key] = function_parameters

    kb_arguments = []
    for argument_type in varseek_ref_only_allowable_kb_arguments:
        arguments_dashes_removed = {argument.lstrip('-').replace('-', '_') for argument in varseek_ref_only_allowable_kb_arguments[argument_type]}
        kb_arguments.extend(arguments_dashes_removed)
    function_parameters_dict_of_sets[kb_name] = set(kb_arguments)

    # Iterate over all combinations of the sets
    for (key1, set1), (key2, set2) in combinations(function_parameters_dict_of_sets.items(), 2):
        shared_elements = set1 & set2  # Find the intersection
        shared_elements = sorted(shared_elements)
        print(f"Shared parameters between {key1} and {key2}: {shared_elements}")

def main():
    function_name_and_key_list_of_tuples_varseek_ref = [(vk.varseek_build.build, "varseek_build"), (vk.varseek_info.info, "varseek_info"), (vk.varseek_filter.filter, "varseek_filter"), (vk.varseek_ref.ref, "varseek_ref")]
    print_all_shared_arguments_between_pairs_of_functions(function_name_and_key_list_of_tuples_varseek_ref, varseek_ref_only_allowable_kb_ref_arguments, kb_name = "kb ref")
    print("\n-----------------------------------\n")
    function_name_and_key_list_of_tuples_varseek_count = [(vk.varseek_fastqpp.fastqpp, "varseek_fastqpp"), (vk.varseek_clean.clean, "varseek_clean"), (vk.varseek_summarize.summarize, "varseek_summarize"), (vk.varseek_count.count, "varseek_count")]
    print_all_shared_arguments_between_pairs_of_functions(function_name_and_key_list_of_tuples_varseek_count, varseek_ref_only_allowable_kb_count_arguments, kb_name = "kb count")

# Run the script if it's executed directly
if __name__ == "__main__":
    main()