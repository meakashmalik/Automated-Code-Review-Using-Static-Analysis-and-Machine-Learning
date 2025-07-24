"""
This is a module for demonstrating Pylint fixes.
"""

def calculate_sum(number1, number2):
    """
    Calculates the sum of two numbers.
    """
    total_result = number1 + number2
    return total_result

# Pylint considers these "constants" if their values don't change
# Convention: Constants should be in UPPER_CASE
FIRST_NUMBER = 10
SECOND_NUMBER = 20

print(calculate_sum(FIRST_NUMBER, SECOND_NUMBER))
