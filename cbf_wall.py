import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class MobileRobotCBF:
    def __init__(self):
        # Physical Input Constraints
        self.v_min, self.v_max = -0.5, 1.5   # m/s
        self.w_min, self.w_max = -1.0, 1.0   # rad/s
        self.u_bounds = [(self.v_min, self.v_max), (self.w_min, self.w_max)] 
        self.alpha_gain = 1.0 # How aggressively to brake

    def get_dynamics(self, theta):
        # f(x) is 0 for kinematic unicycle
        f = np.zeros(3)
        g = np.array([
            [np.cos(theta), 0],
            [np.sin(theta), 0],
            [0, 1]
        ])
        return f, g

    def safety_filter(self, x, u_nominal):
        px, py, theta = x
        
        # Assume Wall is at x = 5.0
        wall_x = 5.0
        h = (wall_x - px)**1
        grad_h = np.array([-1.0, 0.0, 0.0])

        f, g = self.get_dynamics(theta)
        Lf_h = grad_h @ f
        Lg_h = grad_h @ g

        rhs = -self.alpha_gain * h - Lf_h

        def objective(u):
            return np.sum((u - u_nominal)**2)

        cons = ({'type': 'ineq', 'fun': lambda u: Lg_h @ u - rhs})
        
        res = minimize(objective, x0=u_nominal, bounds=self.u_bounds, constraints=cons)
        return res.x if res.success else np.array([0.0, 0.0])

# --- Simulation Setup ---
bot = MobileRobotCBF()
dt = 0.1
steps = 100
state = np.array([0.0, 0.0, 0.0]) # Start at origin, facing right
desired_u = np.array([3.475, 0.0])  # User wants to go 2.0 m/s (above limit!)

trajectory = [state.copy()]
controls = []

# --- Simulation Loop ---
for _ in range(steps):
    # 1. Filter the command
    safe_u = bot.safety_filter(state, desired_u)
    controls.append(safe_u)
    
    # 2. Update Dynamics (Euler Integration)
    f, g = bot.get_dynamics(state[2])
    state_dot = f + g @ safe_u
    state = state + state_dot * dt
    trajectory.append(state.copy())

trajectory = np.array(trajectory)
controls = np.array(controls)

# --- Plotting ---
plt.figure(figsize=(10, 4))

# Plot 1: Top-Down Trajectory
plt.subplot(1, 2, 1)
plt.plot(trajectory[:, 0], trajectory[:, 1], '-b', label='Robot Path')
plt.axvline(x=5.0, color='r', linestyle='--', linewidth=2, label='Obstacle/Wall')
plt.scatter(trajectory[0, 0], trajectory[0, 1], color='g', marker='o', label='Start')
plt.title("Robot Trajectory")
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.legend()
plt.grid(True)

# Plot 2: Velocity over time
plt.subplot(1, 2, 2)
time = np.arange(0, steps * dt, dt)
plt.plot(time, controls[:, 0], '-g', label='Actual Velocity (Filtered)')
plt.axhline(y=1.5, color='orange', linestyle=':', label='Actuator Limit (v_max)')
plt.axhline(y=2.0, color='grey', linestyle='--', label='Desired Velocity')
plt.title("Velocity Profile")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (v)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()