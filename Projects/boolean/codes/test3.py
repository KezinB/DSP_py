import sympy as sp

# Define a function to evaluate the Boolean expression with input values
def evaluate_expression(expression, values):
    try:
        # Substitute the values for the symbols in the expression
        result = expression.subs(values)
        return result
    except Exception as e:
        return f"Error: {str(e)}"

# Simplify the Boolean expression
def simplify_expression(expression):
    return sp.simplify_logic(expression)

# Function to take user input for variable values
def get_input_values(variables):
    values = {}
    print("Enter the values for the variables (1 for True, 0 for False):")
    for var in variables:
        while True:
            try:
                value = int(input(f"Enter value for {var}: "))
                if value not in [0, 1]:
                    print("Invalid input. Please enter 1 for True or 0 for False.")
                else:
                    values[var] = value
                    break
            except ValueError:
                print("Invalid input. Please enter 1 for True or 0 for False.")
    return values

# Function to process and analyze the Boolean expression
def process_expression(expr_str):
    try:
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

# Main function to handle modes and navigation
def boolean_calculator():
    print("Welcome to the Extended Boolean Calculator\n")
    print("You can input a Boolean expression using A, B, C, D, E, F, G, H (e.g., 'A & B | C').")
    print("Choose a mode:")
    print("1. Simplify the expression")
    print("2. Solve the expression by providing input values")
    print("Enter 'exit' to quit.\n")

    while True:
        # User input for the mode
        mode = input("Select mode (1 for Simplify, 2 for Solve, or 'exit' to quit): ").strip()

        if mode.lower() == 'exit':
            print("Exiting the Boolean Calculator.")
            break

        # Input the Boolean expression
        expr_str = input("Enter the Boolean expression: ").strip()

        # Process the expression
        expr = process_expression(expr_str)

        if expr:
            if mode == '1':  # Simplify mode
                simplified_expr = simplify_expression(expr)
                print(f"Simplified Expression: {simplified_expr}")
                solve_choice = input("Do you want to solve the simplified expression with input values? (yes/no): ").strip().lower()
                if solve_choice == 'yes':
                    values = get_input_values(find_variables(simplified_expr))
                    result = evaluate_expression(simplified_expr, values)
                    print(f"Result after solving: {result}")
                else:
                    print("Skipping solving the expression.")
            
            elif mode == '2':  # Solve mode
                # Identify the variables in the input expression
                variables_in_expr = find_variables(expr)
                values = get_input_values(variables_in_expr)
                result = evaluate_expression(expr, values)
                print(f"Result after solving: {result}")

            else:
                print("Invalid mode selected. Please choose 1 or 2.")
        else:
            print("Error in the expression. Please try again.")

# Run the calculator
if __name__ == "__main__":
    boolean_calculator()
