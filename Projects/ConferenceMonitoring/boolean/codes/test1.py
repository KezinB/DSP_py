import sympy as sp

def solve_boolean_expression(expr):
    variables = sp.symbols(' '.join(sorted(list(expr.atoms(sp.Symbol)))))
    solutions = sp.satisfiable(expr, all_models=True)
    return list(solutions)

def simplify_boolean_expression(expr):
    simplified_expr = sp.simplify(expr)
    return simplified_expr

def main():
    print("Select mode:")
    print("1. Solve")
    print("2. Simplify")
    mode = input("Enter 1 or 2: ").strip()

    if mode not in ['1', '2']:
        print("Invalid selection. Please enter 1 for 'Solve' or 2 for 'Simplify'.")
        return

    expression = input("Enter boolean expression: ").strip()
    expr = sp.sympify(expression)

    if mode == '1':
        solutions = solve_boolean_expression(expr)
        print(f"Solutions: {solutions}")
    elif mode == '2':
        simplified_expr = simplify_boolean_expression(expr)
        print(f"Simplified Expression: {simplified_expr}")

if __name__ == "__main__":
    main()
