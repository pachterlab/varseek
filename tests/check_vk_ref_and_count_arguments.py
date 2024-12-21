import inspect
import varseek
import pytest
from varseek.varseek_ref import varseek_ref_only_allowable_kb_ref_arguments
from varseek.varseek_count import varseek_ref_only_allowable_kb_count_arguments

def check_parameter_consistency(ref_func, other_func):
    # Get the signatures of the reference function and the other function
    ref_signature = inspect.signature(ref_func)
    other_signature = inspect.signature(other_func)

    # Convert signatures to dictionaries for easier comparison
    ref_params = ref_signature.parameters
    other_params = other_signature.parameters

    # Loop through each parameter in the other function
    for param_name, other_param in other_params.items():
        # Check if the parameter exists in the reference function
        if param_name not in ref_params:
            print(f"Parameter '{param_name}' in {other_func.__name__} is missing in {ref_func.__name__}.")
            continue

        # Check if the default values match
        ref_param = ref_params[param_name]
        if ref_param.default != other_param.default:
            print(
                f"Default value mismatch for '{param_name}':\n"
                f"  {ref_func.__name__} default: {ref_param.default}\n"
                f"  {other_func.__name__} default: {other_param.default}\n"
            )

def check_kb_parameter_consistency(ref_func, kb_dict_of_sets, kb_name):
    ref_params = inspect.signature(ref_func).parameters
    for argument_type_key in kb_dict_of_sets:
            arguments_dashes_removed = {argument.lstrip('-').replace('-', '_') for argument in kb_dict_of_sets[argument_type_key]}
            for param_name in list(arguments_dashes_removed):
                if param_name not in ref_params:
                    print(f"Parameter '{param_name}' in {kb_name} is missing in {ref_func.__name__}.")

def test_ref_arguments():
    # Reference function (from varseek_ref)
    ref_function = varseek.varseek_ref.ref

    # Functions to check against (from varseek_info)
    functions_to_check = [
        varseek.varseek_build.build,
        varseek.varseek_info.info,
        varseek.varseek_filter.filter
    ]

    # Run the check for each function
    for func in functions_to_check:
        print(f"Checking consistency between {ref_function.__name__} and {func.__name__}...")
        check_parameter_consistency(ref_function, func)
        print("Check complete for varseek.\n")

    # check against kb ref
    print(f"Checking consistency between {ref_function.__name__} and kb ref arguments...")
    check_kb_parameter_consistency(ref_func = ref_function, kb_dict_of_sets = varseek_ref_only_allowable_kb_ref_arguments, kb_name = "kb ref")
    print("Check complete for kb.\n")

def test_count_arguments():
    # Reference function (from varseek_ref)
    ref_function = varseek.varseek_count.count
    
    # Functions to check against (from varseek_info)
    functions_to_check = [
        varseek.varseek_fastqpp.fastqpp,
        varseek.varseek_clean.clean,
        varseek.varseek_summarize.summarize
    ]

    # Run the check for each function
    for func in functions_to_check:
        print(f"Checking consistency between {ref_function.__name__} and {func.__name__}...")
        check_parameter_consistency(ref_function, func)
        print("Check complete.\n")

    # check against kb count
    print(f"Checking consistency between {ref_function.__name__} and kb count arguments...")
    check_kb_parameter_consistency(ref_func = ref_function, kb_dict_of_sets = varseek_ref_only_allowable_kb_count_arguments, kb_name = "kb count")
    print("Check complete for kb.\n")


def main():
    test_ref_arguments()
    test_count_arguments()

# Run the script if it's executed directly
if __name__ == "__main__":
    main()