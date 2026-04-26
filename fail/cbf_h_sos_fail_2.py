import numpy as np
from pydrake.solvers import MathematicalProgram
from pydrake.symbolic import Variables

def demonstrate_joint_bmi_failure():
    print("Attempting to synthesize BOTH h(x) and u(x) simultaneously...")
    
    prog = MathematicalProgram()
    
    # Define our state (Indeterminates)
    state = prog.NewIndeterminates(2, "x")
    px, vx = state[0], state[1]
    
    # 1. Create UNKNOWN h(x)
    # The solver must find the coefficients for this polynomial
    h_poly, _ = prog.NewSosPolynomial(Variables(state), 2)
    h_expr = h_poly.ToExpression()
    
    # 2. Create UNKNOWN u(x)
    # The solver must ALSO find the coefficients for this controller
    c = prog.NewContinuousVariables(2, "c") # Unknown coefficients
    u_expr = c[0]*px + c[1]*vx            # Unknown polynomial u(x)
    
    # System dynamics: dot_px = vx, dot_vx = u(x)
    dh_dpx = h_expr.Differentiate(px)
    dh_dvx = h_expr.Differentiate(vx)
    
    # Calculate h_dot
    # WARNING: dh_dvx contains unknown 'h' coefficients.
    # u_expr contains unknown 'c' coefficients.
    h_dot_expr = dh_dpx * vx + dh_dvx * u_expr
    
    gamma = 1.0
    sos_constraint = h_dot_expr + gamma * h_expr
    
    print("\nFormulating constraint: h_dot + gamma * h >= 0")
    print("Handing constraint to PyDrake...")
    
    try:
        # THE CRASH HAPPENS HERE
        prog.AddSosConstraint(sos_constraint)
        
        # If it somehow works (it won't), we would solve it here
        # solver.Solve(prog) 
        
    except Exception as e:
        print("\n" + "!"*60)
        print(" FATAL BMI ERROR CAUGHT!")
        print("!"*60)
        print("\nPyDrake crashed with the following error:")
        print(f"--> {str(e).split('is non-linear')[0]} ... is non-linear.")
        print("\nCONCLUSION:")
        print("You cannot jointly synthesize h(x) and u(x).")
        print("The solver sees unknown variables multiplied by unknown variables,")
        print("which makes the optimization Non-Convex (a BMI).")
        print("!"*60)

if __name__ == "__main__":
    demonstrate_joint_bmi_failure()