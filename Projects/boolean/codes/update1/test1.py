import RPi.GPIO as GPIO
import time
import sympy as sp

# Define the GPIO pins in a 4x4 matrix layout
button_matrix = [
    [40, 32, 38, 36],  # A, B, C, D
    [33, 37, 31, 29],  # ., +, !, XNOR
    [12, 35, 7, 8],    # 1, 0, (, )
    [10, 16, 15, 13]   # CLR, CANCEL, 3, ENTER
]

# Map buttons to their corresponding symbols/operations
button_map = {
    (0, 0): 'A', (0, 1): 'B', (0, 2): 'C', (0, 3): 'D',
    (1, 0): '.', (1, 1): '+', (1, 2): '!', (1, 3): 'XNOR',
    (2, 0): '1', (2, 1): '0', (2, 2): '(', (2, 3): ')',
    (3, 0): 'CLR', (3, 1): 'CANCEL', (3, 2): '3', (3, 3): 'ENTER'
}

# Flatten the list to get all pins
button_pins = [pin for row in button_matrix for pin in row]

# Set up GPIO
GPIO.setmode(GPIO.BOARD)  # Use board pin numbering

# Configure each pin as an input with a pull-up resistor
for pin in button_pins:
    GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Function to evaluate the Boolean expression with input values
def evaluate_expression(expression, values):
    try:
        # Substitute the values for the symbols in the expression
        result = expression.subs(values)
        return result
    except Exception as e:
        return f"Error: {str(e)}"

# Simplify the Boolean expression
def simplify_expression(expression):
    return sp.simplify_logic(expression, form='dnf')

# Function to take user input for variable values using buttons
def get_input_values(variables):
    values = {}
    print("Enter the values for the variables (1 for True, 0 for False):")
    for var in variables:
        while True:
            print(f"Press 1 (True) or 0 (False) for {var}:")
            value = None
            while value is None:
                for i, row in enumerate(button_matrix):
                    for j, pin in enumerate(row):
                        if GPIO.input(pin) == GPIO.LOW:
                            symbol = button_map.get((i, j), "?")
                            if symbol in ['1', '0']:
                                value = int(symbol)
                                break
                    if value is not None:
                        break
                time.sleep(0.2)
            values[var] = bool(value)
            print(f"{var} = {value}")
            break
    return values

# Function to process and analyze the Boolean expression
def process_expression(expr_str):
    try:
        # Replace the custom operators with sympy compatible operators
        expr_str = expr_str.replace('.', ' & ').replace('+', ' | ').replace('!', '~')
        # Convert input string into a sympy Boolean expression
        expr = sp.sympify(expr_str, evaluate=False)
        return expr
    except Exception as e:
        print(f"Error in expression parsing: {e}")
        return None

# Function to find variables in the expression
def find_variables(expression):
    # Extract all symbols (variables) used in the expression
    return list(expression.free_symbols)

# Function to read button input for expression and mode selection
def read_buttons():
    while True:
        print("Welcome to the Extended Boolean Calculator\n")
        print("You can input a Boolean expression using A, B, C, D, E, F, G, H (e.g., 'A . B + C').")
        print("Choose a mode:")
        print("1. Simplify the expression")
        print("2. Solve the expression by providing input values")
        print("Press CANCEL to exit.\n")

        # Mode selection
        mode = None
        while mode is None:
            for i, row in enumerate(button_matrix):
                for j, pin in enumerate(row):
                    if GPIO.input(pin) == GPIO.LOW:
                        symbol = button_map.get((i, j), "?")
                        if symbol == '1':
                            mode = '1'
                            print("Mode selected: Simplify")
                        elif symbol == '3':
                            mode = '2'
                            print("Mode selected: Solve")
                        elif symbol == 'CANCEL':
                            print("Exiting the Boolean Calculator.")
                            return
            time.sleep(0.2)

        # Input the Boolean expression
        input_buffer = ""
        print("Enter the Boolean expression (press ENTER to confirm):")
        while True:
            for i, row in enumerate(button_matrix):
                for j, pin in enumerate(row):
                    if GPIO.input(pin) == GPIO.LOW:
                        symbol = button_map.get((i, j), "?")
                        if symbol == 'ENTER':
                            print(f"\nThe entered expression is: {input_buffer}\n")
                            expr = process_expression(input_buffer)
                            if expr:
                                if mode == '1':  # Simplify mode
                                    simplified_expr = simplify_expression(expr)
                                    print(f"Simplified Expression: {simplified_expr}")
                                    print("Press 1 to solve the simplified expression or CANCEL to skip.")
                                    solve_choice = None
                                    while solve_choice is None:
                                        for i, row in enumerate(button_matrix):
                                            for j, pin in enumerate(row):
                                                if GPIO.input(pin) == GPIO.LOW:
                                                    symbol = button_map.get((i, j), "?")
                                                    if symbol == '1':
                                                        solve_choice = 'yes'
                                                    elif symbol == 'CANCEL':
                                                        solve_choice = 'no'
                                        time.sleep(0.2)
                                    if solve_choice == 'yes':
                                        values = get_input_values(find_variables(simplified_expr))
                                        result = evaluate_expression(simplified_expr, values)
                                        result = 1 if result else 0  # Convert True/False to 1/0
                                        print(f"Result after solving: {result}")
                                    else:
                                        print("Skipping solving the expression.")
                                elif mode == '2':  # Solve mode
                                    values = get_input_values(find_variables(expr))
                                    result = evaluate_expression(expr, values)
                                    result = 1 if result else 0  # Convert True/False to 1/0
                                    print(f"Result after solving: {result}")
                            break
                        elif symbol == 'CLR':
                            input_buffer = ""
                            print("\nInput cleared.\nInput: ", end="", flush=True)
                        elif symbol == 'CANCEL':
                            input_buffer = input_buffer[:-1]
                            print(f"\rInput: {input_buffer} ", end="", flush=True)
                        else:
                            input_buffer += symbol
                            print(f"{symbol} ", end="", flush=True)
                time.sleep(0.2)
            break

try:
    read_buttons()
except KeyboardInterrupt:
    print("\nExiting...")
finally:
    GPIO.cleanup()  # Cleanup GPIO settings before exit