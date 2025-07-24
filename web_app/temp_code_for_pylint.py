# This is my trial code for the automated review tool
import os # This import is not used later in the code

def MyFunction(first_num, second_num): # Function name is not snake_case, args not used
    # This is a comment that Pylint usually ignores, but it's part of the line
    result_value = first_num + second_num
    return result_value

global_variable = 10
another_global_variable = 20 # Name not UPPER_CASE for a constant

# A very long line to trigger line-too-long warning:
this_is_a_very_very_long_line_of_code_that_will_definitely_exceed_the_100_character_limit_for_pylint_line_length_checking = 1

# Calling the function
print(MyFunction(global_variable, another_global_variable))