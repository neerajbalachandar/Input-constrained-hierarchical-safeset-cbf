import numpy as np
from pydrake.solvers import MathematicalProgram
from pydrake.symbolic import Variables

def demonstrate_joint_bmi_failure():
    print("Attempting to synthesize BOTH h(x) and u(x) simultaneously...")
    
    prog = MathematicalProgram()
    
    #state defn
    state = prog.NewIndeterminates(2, "x")
    px, vx = state[0], state[1]
    
    #UNKNOWN h(x)
    h_poly, _ = prog.NewSosPolynomial(Variables(state), 2)
    h_expr = h_poly.ToExpression()
    
    #UNKNOWN u(x)
    c = prog.NewContinuousVariables(2, "c") 
    u_expr = c[0]*px + c[1]*vx            #polynomial
    
    
    dh_dpx = h_expr.Differentiate(px)
    dh_dvx = h_expr.Differentiate(vx)
    
 
    h_dot_expr = dh_dpx * vx + dh_dvx * u_expr
    
    gamma = 1.0
    sos_constraint = h_dot_expr + gamma * h_expr
    
    print("\nFormulating constraint: h_dot + gamma * h >= 0")
    
    try:
       
        prog.AddSosConstraint(sos_constraint) #may be infeasible
        
        # solver.Solve(prog) 
        
    except Exception as e:
        print("\n" + "!"*60)
        print(" FATAL BMI ERROR CAUGHT!")
        print("!"*60)
        print("\nPyDrake crashed with the following error:")
        print(f"--> {str(e).split('is non-linear')[0]} ... is non-linear.")
        print("\nCONCLUSION:")
        print("The solver sees unknown variables multiplied by unknown variables,")
        print("which makes the optimization Non-Convex (a BMI).")
        print("!"*60)

if __name__ == "__main__":
    demonstrate_joint_bmi_failure()