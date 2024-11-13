import inspect
import varseek
import pytest

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
        print("Check complete.\n")

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


def main():
    test_ref_arguments()
    test_count_arguments()

# Run the script if it's executed directly
if __name__ == "__main__":
    main()