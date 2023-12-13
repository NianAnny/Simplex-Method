import numpy as np

# Allow LP to be in standard form, start at origin
# If it is a maximization problem, directly follow the instruction
# If it is a minimization problem, multiply coefficients of objective by -1

# Simplex Method Algorithm (Maximization Problem)
def Simplex_Method(c, A, b):
    print("\n===============================") # a signal for solving a new problem
    # Parameters:
    # c: objective vector (n * 1)
    # A: coefficient constraint matrix (m * n)
    # b: constraint vector (m * 1)

    # Wants to return the optimal objective value and the solution vector

    # Check that inputs have consistent dimensions
    m, n = A.shape  # count number of constraints (m), and number of variables (n)
    if len(b) != m or len(c) != n:
        return ("The input is inconsistent.")

    # Check the feasibility of initial solution at origin
    x_origin = np.array(np.zeros(n)) # assume all variables are zeros
    Ax = A @ x_origin
    if np.any(Ax > b):
         return (f"The initial solution at {x_origin} is infeasible!")

    # Add slack/surplus variables forming the canonical form
    c_s = np.hstack([c, np.zeros(m)]) # add 0s as slack coefficient
    A_s = np.hstack([A, np.identity(m)]) # add an identity matrix for slacks

    # Check for Basis and nonBasis indices in coefficient matrix
    # Convert the standard form to canonical form
    basicVars = list(range(n, n + m)) # slacks' indices
    nonBasicVars = list(range(n)) # variables' indices

    # Initialization
    # create x vector as the same length as that of n variables including slacks (n+m)
    x = np.zeros(n + m)
    x[basicVars] = b  # construct a vector contains all variables values so far

    iteration = 100 # set up max iteration to avoid infinite loops

    for i in range(iteration):
        # Ensure order of variables is correct
        basicVars = sorted(basicVars)
        nonBasicVars = sorted(nonBasicVars)

        B = A_s[:, basicVars] # basis of coefficient matrix
        N = A_s[:, nonBasicVars] # nonBasis of coefficient matrix
        cB = c_s[basicVars] # basic objective vector
        cN = c_s[nonBasicVars] # nonBasic objective vector
        xB = x[basicVars] # basic variable values
        b = xB # define basic variable vector

        # Step 1: Compute simplex multipliers (y) and reduced costs (cNbar)
        y = np.linalg.solve(np.transpose(B), cB)
        cNbar = cN - np.transpose(y) @ N

        # Check for the optimality
        if np.all(cNbar <= 0): # reach the optimum
            break

        # Step 2: Compute the reduced cost
        if np.any(cNbar > 0):
            # Store the results of the current iteration
            x = x  # store basic variable values, nonBasic ones are 0

        # Step 3: Compute simplex direction
        entering_index = np.argmax(cNbar) # check for the index of the variable direction we will choose
        entering = nonBasicVars[entering_index] # the entering variable index
        dB = np.linalg.solve(B, -A_s[:, entering]) # direction

        # Check for unboundedness
        if np.all(dB >= 0):
            return ("Detect that the problem is unbounded!")

        print(f"Iteration_{i}:")
        print(f"Start at {x}")  # start point for iteration_i

        # Step 4: Compute maximum step size
        steps = np.where(dB < 0, -b/dB, np.inf) # available step sizes,
        step_max = np.min(steps) # select the max step value

        leaving_index = np.argmin(steps) # define the leaving variable's index
        leaving = basicVars[leaving_index] # define the leaving variable

        # Step 5: Update the solution and basis
        b = b + step_max * dB # basic variable values change
        x[sorted(basicVars)] = b # current values for basic variables
        b[leaving_index] = step_max  # new basic variable values

        # replace leaving variable with nonBasic entering and entering with leaving in the solution vector
        basicVars[leaving_index] = entering
        nonBasicVars[entering_index] = leaving

        # Store the results of the current iteration, renew the solution
        x[basicVars] = b
        optimal_value = c_s @ x # calculate the optimal value

        # Check B, N, cB, cN, xB for the current iteration
        #print(f"B: {B}")
        #print(f"N: {N}")
        #print(f"cB: {cB}")
        #print(f"cN: {cN}")
        #print(f"xB: {xB}")

        # check for reduced cost, entering variable, direction, max step for the current iteration
        #print(f"reduced cost (cNbar): {cNbar}, entering: cN_{entering}")
        #print(f"dB: {dB}")
        #print(f"lambda: {steps}, select lambda_max: {step_max}")

        # Check for objective value and point for the current iteration
        print(f"Current point: {x}")
        print(f"Objective value: {optimal_value}")
        print("\n-------------------------------") # a signal of another iteration or the solution

        # Return the optimal value and the corresponding point
    return {"Optimal value": optimal_value,
            "Solution vector": x}

def ask_input():
    # Type of problem
    while True:
        problem_type = input("max/min problem? (Enter 'max' or 'min'):\n").strip().lower()
        if problem_type == 'max' or problem_type == 'min':
            break
        else:
            print("Invalid, please enter 'max' or 'min'.")

    # Remind the problem should be in standard form
    print(f"\nLP should be in standard form:")

    if problem_type == 'max':
        print(f"{problem_type} c^T x")
        print(f"s.t. Ax <= b")
        print(f"      x >= 0")
    else:
        print(f"{problem_type} c^T x ---> max -c^T x")
        print(f"               s.t. Ax <= b")
        print(f"                     x >= 0")

    # Ask for number of variables and constraints
    n = int(input("Number of variables (n):\n"))
    m = int(input("Number of constraints (m):\n"))

    # Get the objective function coefficients
    while True:
        try:
            c = np.array(list(map(float, input(f"Coefficients of the objective (separate # by comma) for {n} variables:\n").split(","))))
            if len(c) != n:
                raise ValueError
            break
        except ValueError:
            print(f"\nPlease enter exactly {n} numbers separated by ','.")

    # Coefficient matrix of constraints (A)
    A = []
    for i in range(m):
        while True:
            try:
                row = np.array(list(map(float, input(f"Coefficients for constraint {i+1} (separated by comma) for {n} variables:\n").split(","))))
                if len(row) != n:
                    raise ValueError
                A.append(row)
                break
            except ValueError:
                print(f"\nPlease enter exactly {n} numbers (separated by ',').")
    A = np.array(A)

    # Constraint vector
    while True:
        try:
            b = np.array(list(map(float, input(f"RHS (b) values (separated by comma) for the {m} constraints:\n").split(","))))
            if len(b) != m:
                raise ValueError
            break
        except ValueError:
            print(f"\nPlease enter exactly {m} numbers (separated by ',').")

    return problem_type, c, A, b

# Call the function to get inputs
problem_type, c, A, b = ask_input()

# Call the Simplex_Method
outcome = Simplex_Method(c, A, b)
print(outcome)