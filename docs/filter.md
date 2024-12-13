Takes in:
- info csv
- mcrs fasta
- filters in the following format

COLUMN-RULE=VALUE
- COLUMN: column name in the info csv
- RULE: rule to apply to the column from the following list:
    - min: minimum value
    - max: maximum value
    - between: between two values (inclusive)
    - toppercent: top x% of values (eg 1 to keep top 1%)
    - bottompercent: bottom x% of values (eg 1 to keep bottom 1%)
    - equal: equal to value
    - notequal: not equal to value
    - isin: equals an element in value (set, or a txt file where each value is separated by a new line)
    - isnotin: does not equal an element in value (set, or a txt file where each value is separated by a new line)
    - istrue: is True
    - isfalse: is False
    - isnottrue: is not True (i.e., False or NaN)
    - isnotfalse: is not False (i.e., True or NaN)
    - isnull: is null
    - isnotnull: is not null
- VALUE: value to compare to
    - min, max: single numeric value (e.g., 1)
    - between: two numeric values separated by a comma (e.g., "1,2")
    - contains, notcontains: one or more numeric values separated by a comma (e.g., "1,2,3") for command line or python, or a list or set passed in through an fstring (python only)
    - equal, notequal: single value (e.g., "yes")
    - istrue, isfalse, isnottrue, isnotfalse, isnull, isnotnull: no value needed

on command line, simply list the filters as the last argument; in python, pass them as a list of strings

OR the filters can be passed in as a txt file - example:
COLUMN1-RULE1=VALUE1
COLUMN2-RULE2=VALUE2
COLUMN3-RULE3=VALUE3

While the order of filters does not affect the output filtered fasta file, it will affect the stats printed to the console when in verbose mode. The stats will be printed in the order of the filters.