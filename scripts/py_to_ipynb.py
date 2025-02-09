import argparse

import nbformat as nbf

# Path to your script
python_in_path = '/home/jrich/Desktop/varseek/notebooks/test_module.py'
notebook_out_path = '/home/jrich/Desktop/varseek/notebooks/test_module.ipynb'
comment_marker_deliniator = '# CELL'

def module_to_notebook(python_in_path, notebook_out_path, comment_marker_deliniator):
    # Read your script
    with open(python_in_path, 'r', encoding="utf-8") as f:
        lines = f.readlines()

    # Initialize notebook
    nb = nbf.v4.new_notebook()
    cells = []

    # Initialize variables
    imports_and_assignments = []
    parameters = {}
    function_code = []
    in_function = False
    function_def_line = ''

    for line in lines:
        stripped_line = line.strip()
        if not in_function:
            if stripped_line.startswith('def '):
                in_function = True
                function_def_line = stripped_line
                # Extract parameters from function definition line
                params_str = function_def_line[function_def_line.find('(')+1:function_def_line.find(')')]
                params_list = params_str.split(',')
                for param in params_list:
                    param = param.strip()
                    if '=' in param:
                        param_name, param_value = param.split('=',1)
                        param_name = param_name.strip()
                        param_value = param_value.strip()
                        parameters[param_name] = param_value
                    else:
                        parameters[param.strip()] = None  # No default value
            else:
                imports_and_assignments.append(line)
        else:
            # We are inside the function
            function_code.append(line)

    # Remove one level of indentation from function code
    processed_function_code = []
    for line in function_code:
        if line.startswith('    '):
            processed_line = line[4:]
        else:
            processed_line = line
        processed_function_code.append(processed_line)

    # Create first cell: imports and assignments
    imports_cell = ''.join(imports_and_assignments).rstrip()  # Remove trailing spaces/newlines
    cells.append(nbf.v4.new_code_cell(imports_cell))

    # Create second cell: parameter defaults
    param_assignments = ''
    for param_name, param_value in parameters.items():
        if param_value is not None:
            param_assignments += f"{param_name} = {param_value}\n"
        else:
            param_assignments += f"{param_name} = None\n"
    cells.append(nbf.v4.new_code_cell(param_assignments.rstrip()))  # Remove trailing newlines

    # Split function code into cells at '# CELL' markers
    current_cell_code = []
    for line in processed_function_code:
        if line.strip() == comment_marker_deliniator:
            if current_cell_code:
                while current_cell_code and current_cell_code[-1].strip() == '':
                    current_cell_code.pop()
                # Join the current code into one block and strip trailing newlines
                cell_code = ''.join(current_cell_code).rstrip()
                cells.append(nbf.v4.new_code_cell(cell_code))
                current_cell_code = []
        else:
            current_cell_code.append(line)

    # Add the last cell
    if current_cell_code:
        cell_code = ''.join(current_cell_code)
        cells.append(nbf.v4.new_code_cell(cell_code))

    # Set notebook cells
    nb['cells'] = cells

    # Write notebook to file
    with open(notebook_out_path, 'w', encoding="utf-8") as f:
        nbf.write(nb, f)


if __name__ == "__main__":
    # Set up argparse to handle input arguments
    parser = argparse.ArgumentParser(description="Convert a Python script to a Jupyter notebook.")
    parser.add_argument('-i', '--input', required=True, help="Path to the input Python script")
    parser.add_argument('-o', '--output', help="Path to the output Jupyter notebook")
    parser.add_argument('-d', '--delimiter', default='# CELL', help="Comment marker for cell splits (default: '# CELL')")

    # Parse the arguments
    args = parser.parse_args()

    if args.output is None:
        args.output = args.input.replace('.py', '.ipynb')

    # Call the conversion function with parsed arguments
    module_to_notebook(args.input, args.output, args.delimiter)
