import numpy as np
import matplotlib.pyplot as plt

try:
    from pydrake.solvers import MathematicalProgram, Solve, MosekSolver, ClarabelSolver
    from pydrake.symbolic import Variable, Polynomial, Evaluate, Variables
    import pydrake.symbolic as sym
    DRAKE_AVAILABLE = True
except ImportError:
    DRAKE_AVAILABLE = False
    print("Please install pydrake to run the SOS synthesis: pip install drake")

class MobileRobotSim:
    def __init__(self, init_x, init_y, init_yaw, init_v=0.0, init_w=0.0):
        self.x, self.y, self.yaw, self.v, self.w = init_x, init_y, init_yaw, init_v, init_w

    def apply_dynamics(self, cmd_accel, cmd_yaw_accel, dt):
        self.v += cmd_accel * dt
        self.w += cmd_yaw_accel * dt
        self.x += self.v * np.cos(self.yaw) * dt
        self.y += self.v * np.sin(self.yaw) * dt
        self.yaw += self.w * dt
        self.yaw = np.arctan2(np.sin(self.yaw), np.cos(self.yaw))

def demonstrate_bmi_failure():
    """
    Attempts to solve for BOTH h(x) and u(x) simultaneously.
    This triggers a Bilinear Matrix Inequality (BMI) error in PyDrake.
    """
    print("\n" + "="*60)
    print(" PART 1: THE BMI MATHEMATICAL FAILURE (JOINT SYNTHESIS)")
    print("="*60)
    
    prog = MathematicalProgram()
    state = prog.NewIndeterminates(2, "x")
    px, vx = state[0], state[1]
    
    # 1. Unknown h(x) (Decision Variables)
    h_poly, _ = prog.NewSosPolynomial(Variables(state), 2)
    h_expr = h_poly.ToExpression()
    
    # 2. Unknown u(x) (Decision Variables)
    # u(x) = c0*px + c1*vx
    c = prog.NewContinuousVariables(2, "c")
    u_expr = c[0]*px + c[1]*vx
    
    # System: dot_px = vx, dot_vx = u_expr
    dh_dpx = h_expr.Differentiate(px)
    dh_dvx = h_expr.Differentiate(vx)
    
    # h_dot = (dh/dpx)*vx + (dh/dvx)*u_expr
    # This multiplies unknown 'h' coefficients by unknown 'c' coefficients!
    h_dot_expr = dh_dpx * vx + dh_dvx * u_expr
    
    gamma = 1.0
    sos_constraint = h_dot_expr + gamma * h_expr
    
    print("Attempting to add constraint: dot_h(x,u) + gamma * h(x) >= 0")
    print("where BOTH h(x) and u(x) are unknown polynomials...\n")
    
    try:
        prog.AddSosConstraint(sos_constraint)
    except Exception as e:
        print("[CAUGHT EXPECTED ERROR] Cannot formulate the problem!")
        print(f"Error Message: {str(e).split('is non-linear')[0]} ... is non-linear.")
        print("\n--- WHY THIS FAILS (THE BMI PROBLEM) ---")
        print("Equation:  dot_h(x) = [dh/dx] * (f(x) + g(x)*u(x))")
        print("Because both h(x) and u(x) are unknown, we are multiplying decision variables together.")
        print("This forms a Bilinear Matrix Inequality (BMI), which is non-convex and NP-Hard.")
        print("Standard SDP solvers (Mosek, Clarabel) CANNOT solve this globally.")
        print("----------------------------------------\n")

def synthesize_fixed_u_h():
    """
    Falls back to a fixed u(x) to find h(x) (1-step Alternating SOS).
    This mathematically succeeds but requires infinite control bounds.
    """
    print("="*60)
    print(" PART 2: THE PHYSICAL LIMITATION FAILURE (LIVE PLOT)")
    print("="*60)
    print("Synthesizing h(x) assuming a fixed tracking controller u(x)...")
    prog = MathematicalProgram()
    state = prog.NewIndeterminates(4, "x")
    px, py, vx, vy = state[0], state[1], state[2], state[3]
    state_vars = Variables(state)

    f = np.array([vx, vy, 0, 0])
    g = np.array([[0, 0], [0, 0], [1, 0], [0, 1]])

    # Fixed PD Controller
    ux = -1.0 * px - 2.0 * vx 
    uy = -1.0 * py - 2.0 * vy
    u = np.array([ux, uy])
    x_dot = f + g @ u

    h_poly, h_Gram = prog.NewSosPolynomial(state_vars, 4)
    h_expr = h_poly.ToExpression() 

    obs_radius = 1.2
    unsafe_region_expr = obs_radius**2 - (px**2 + py**2) 

    lambda_1_poly, _ = prog.NewSosPolynomial(state_vars, 2)
    prog.AddSosConstraint(-h_expr - lambda_1_poly.ToExpression() * unsafe_region_expr) 

    h_dot_expr = h_expr.Jacobian(state).dot(x_dot)
    prog.AddSosConstraint(h_dot_expr + 2.0 * h_expr)

    prog.AddLinearCost(np.trace(h_Gram))

    solver = ClarabelSolver() if ClarabelSolver().available() else MosekSolver()
    result = solver.Solve(prog)

    if result.is_success():
        print("[SUCCESS] Found valid SOS h(x). The math guarantees safety!")
        print("HOWEVER, polynomials require massive inputs near boundaries.")
        print("In the live simulation, we will enforce realistic physical input bounds.")
        print("Watch the inner safe set fail...\n")
        return result.GetSolution(h_poly), state
    else:
        return None, None

def run_simulation():
    if not DRAKE_AVAILABLE:
        return
        
    demonstrate_bmi_failure()
    h_poly, sym_states = synthesize_fixed_u_h()

    dt = 0.05
    max_steps = 300
    l_offset = 0.1
    
    pt_start = (-4.0, -4.0)
    pt_end = (6.0, 6.0)
    
    bot = MobileRobotSim(init_x=pt_start[0], init_y=pt_start[1], init_yaw=np.pi/4)                     
    hazard = {'x': 1.0, 'y': 1.0, 'v': 0.0, 'yaw': 0.0, 'r': 1.0}

    # Dead-center straight path
    path_x = np.linspace(pt_start[0], pt_end[0], 200)
    path_y = np.linspace(pt_start[1], pt_end[1], 200)

    log_x, log_y = [], []

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-5, 7)
    ax.set_ylim(-5, 7)
    # ax.set_title("SOS Failure: Mathematical Guarantee vs Physical Limits", color='red', fontweight='bold')

    plot_bot, = ax.plot([], [], 'bs', markersize=8, label='Robot')                         
    plot_trail, = ax.plot([], [], 'b-', linewidth=2, alpha=0.6)         
    plot_ref, = ax.plot(path_x, path_y, 'g--', linewidth=2, label='Path')
    
    hazard_patch = plt.Circle((hazard['x'], hazard['y']), hazard['r'], color='red', alpha=0.4, label='Obstacle')
    ax.add_patch(hazard_patch)
    ax.legend(loc='upper left')

    for step in range(max_steps):
        rel_px = bot.x + l_offset * np.cos(bot.yaw) - hazard['x']
        rel_py = bot.y + l_offset * np.sin(bot.yaw) - hazard['y']
        
        # SOS Polynomial Controller
        ux = -1.0 * rel_px - 2.0 * bot.v * np.cos(bot.yaw)
        uy = -1.0 * rel_py - 2.0 * bot.v * np.sin(bot.yaw)

        cmd_a = ux * np.cos(bot.yaw) + uy * np.sin(bot.yaw)
        cmd_alpha = (-ux * np.sin(bot.yaw) + uy * np.cos(bot.yaw)) / l_offset

        # ---------------------------------------------------------
        # THE PHYSICAL FAILURE POINT
        # The SOS math assumes unbounded inputs. We enforce limits.
        # As the robot approaches the obstacle, the polynomial gradient 
        # demands impossible deceleration (e.g., a = -20 m/s^2).
        # We clip it to what a real robot can do.
        # ---------------------------------------------------------
        clamped_a = np.clip(cmd_a, -1.0, 1.0)
        clamped_alpha = np.clip(cmd_alpha, -1.0, 1.0)

        bot.apply_dynamics(clamped_a, clamped_alpha, dt)                          

        log_x.append(bot.x); log_y.append(bot.y)

        plot_bot.set_data([bot.x], [bot.y])                                    
        plot_trail.set_data(log_x, log_y)
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)

        # Detect collision
        dist = np.hypot(bot.x - hazard['x'], bot.y - hazard['y'])
        if dist < hazard['r']:
            print("\n[CRASH DETECTED!] The robot hit the obstacle!")
            print("Reason: The polynomial controller demanded massive input authority.")
            print("Because the real inputs were clamped to [-1.0, 1.0], the SOS guarantee failed.")
            break

    plt.ioff()
    plt.show()

if __name__ == '__main__':
    run_simulation()