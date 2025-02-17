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

def process_expression(expr_str):
    try:
        expr_str = expr_str.replace('.', ' & ').replace('+', ' | ').replace('!', '~')
        return sp.sympify(expr_str, evaluate=False)
    except Exception as e:
        print(f"Error in expression parsing: {e}")
        return None

def read_buttons():
    while True:
        mode = ""
        print("Select mode: 0 for Simplify, 1 for Solve")
        while True:
            for i, row in enumerate(button_matrix):
                for j, pin in enumerate(row):
                    if GPIO.input(pin) == GPIO.LOW:
                        symbol = button_map.get((i, j), "?")
                        if symbol in ['0', '1']:
                            mode = symbol
                            print(f"Mode selected: {'Simplify' if mode == '0' else 'Solve'}")
                            break
                if mode:
                    break
            if mode:
                break
            time.sleep(0.35)
        
        input_buffer = ""
        print("Input:", end=" ", flush=True)
        while True:
            for i, row in enumerate(button_matrix):
                for j, pin in enumerate(row):
                    if GPIO.input(pin) == GPIO.LOW:
                        symbol = button_map.get((i, j), "?")
                        if symbol == 'ENTER':
                            print(f"\nThe entered expression is: {input_buffer}\n")
                            expr = process_expression(input_buffer)
                            if expr:
                                if mode == '0':  # Simplify mode
                                    simplified_expr = sp.simplify_logic(expr, form='dnf')
                                    print(f"Simplified Expression: {simplified_expr}")
                                else:  # Solve mode
                                    variables = list(expr.free_symbols)
                                    if variables:
                                        values = {}
                                        print("Enter values for:")
                                        for var in variables:
                                            while True:
                                                val = input(f"{var} (0/1): ")
                                                if val in ['0', '1']:
                                                    values[var] = int(val)
                                                    break
                                                print("Invalid input, enter 0 or 1.")
                                        result = expr.subs(values)
                                        print(f"Result: {int(result)}")
                            return  # Return to mode selection after processing
                        elif symbol == 'CLR':
                            input_buffer = ""
                            print("\nInput cleared.\nInput: ", end="", flush=True)
                        elif symbol == 'CANCEL':
                            input_buffer = input_buffer[:-1]
                            print(f"\rInput: {input_buffer} ", end="", flush=True)
                        else:
                            input_buffer += symbol
                            print(f"{symbol} ", end="", flush=True)
            time.sleep(0.35)

try:
    read_buttons()
except KeyboardInterrupt:
    print("\nExiting...")
    GPIO.cleanup()  # Cleanup GPIO settings before exit