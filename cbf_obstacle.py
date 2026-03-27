import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ================= ROBOT MODEL =================
class MobileRobot:
    def __init__(self, v_min=-0.5, v_max=0.5, w_min=-1.0, w_max=1.0):
        self.v_min = v_min
        self.v_max = v_max
        self.w_min = w_min
        self.w_max = w_max
        self.u_bounds = [(self.v_min, self.v_max), (self.w_min, self.w_max)]

    def get_dynamics(self, theta):
        f = np.zeros(3)
        g = np.array([
            [np.cos(theta), 0.0],
            [np.sin(theta), 0.0],
            [0.0,           1.0]
        ])
        return f, g


# ================= OBSTACLE + BARRIER =================
x0, y0, r0 = 5.0, 0.0, 1.0
alpha_gain = 5.0

def barrier_function(x):
    px, py, theta = x
    h = (px - x0)**2 + (py - y0)**2 - r0**2
    grad_h = np.array([2*(px-x0), 2*(py-y0), 0.0])
    return h, grad_h


# ================= SAFETY FILTER =================
def safety_filter(robot, x, u_nominal):
    h, grad_h = barrier_function(x)
    theta = x[2]
    f, g = robot.get_dynamics(theta)

    Lf_h = grad_h @ f
    Lg_h = grad_h @ g
    rhs = -alpha_gain * h - Lf_h

    def objective(u):
        return np.sum((u - u_nominal)**2)

    cons = ({'type': 'ineq', 'fun': lambda u: Lg_h @ u - rhs})

    res = minimize(objective, x0=u_nominal,
                   bounds=robot.u_bounds,
                   constraints=cons)

    return res.x if res.success else np.array([0.0, 0.0])


# ================= NOMINAL CONTROLLER =================
def nominal_controller(state):
    # intentionally unsafe (no obstacle awareness)
    return np.array([0.5, 0.0])


# ================= SIMULATION =================
def simulate(robot, use_cbf=True, dt=0.1, steps=120):
    state = np.array([0.0, 0.0, 0.0])

    traj, h_vals = [state.copy()], []
    u_nom_hist, u_app_hist = [], []

    min_h = np.inf
    unsafe_steps = 0

    for _ in range(steps):
        u_nom = nominal_controller(state)

        if use_cbf:
            u_app = safety_filter(robot, state, u_nom)
        else:
            u_app = np.array([
                np.clip(u_nom[0], robot.v_min, robot.v_max),
                np.clip(u_nom[1], robot.w_min, robot.w_max)
            ])

        h, _ = barrier_function(state)

        if h < 0:
            unsafe_steps += 1

        min_h = min(min_h, h)

        f, g = robot.get_dynamics(state[2])
        state = state + (f + g @ u_app) * dt

        traj.append(state.copy())
        h_vals.append(h)
        u_nom_hist.append(u_nom)
        u_app_hist.append(u_app)

    return {
        "traj": np.array(traj),
        "h": np.array(h_vals),
        "u_nom": np.array(u_nom_hist),
        "u_app": np.array(u_app_hist),
        "time": np.arange(0, steps*dt, dt),
        "min_h": min_h,
        "unsafe": unsafe_steps
    }


# ================= VISUALIZATION =================
def plot_trajectory(results, title):
    traj = results["traj"]

    x_vals = np.linspace(-1, 8, 400)
    y_vals = np.linspace(-3, 3, 400)
    X, Y = np.meshgrid(x_vals, y_vals)
    H = (X-x0)**2 + (Y-y0)**2 - r0**2

    plt.figure(figsize=(7,6))

    plt.contourf(X, Y, H, levels=60, cmap='coolwarm')
    plt.contour(X, Y, H, levels=[0], colors='black', linewidths=2)

    plt.plot(traj[:,0], traj[:,1], 'b', linewidth=2)
    plt.scatter(traj[0,0], traj[0,1], c='g', label='Start')
    plt.scatter(traj[-1,0], traj[-1,1], c='m', label='End')

    theta = np.linspace(0, 2*np.pi, 200)
    plt.fill(x0 + r0*np.cos(theta), y0 + r0*np.sin(theta),
             color='red', alpha=0.3)

    plt.title(title)
    plt.axis('equal')
    plt.grid()
    plt.legend()
    plt.show()


def plot_barrier(results, title):
    plt.figure(figsize=(6,4))
    plt.plot(results["time"], results["h"])
    plt.axhline(0, linestyle='--')
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("h(x)")
    plt.grid()
    plt.show()


def plot_controls(results, robot, title):
    t = results["time"]
    u_nom = results["u_nom"]
    u_app = results["u_app"]

    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.plot(t, u_nom[:,0], '--k')
    plt.plot(t, u_app[:,0], 'g')
    plt.axhline(robot.v_max, linestyle=':')
    plt.axhline(robot.v_min, linestyle=':')
    plt.title("v")

    plt.subplot(1,2,2)
    plt.plot(t, u_nom[:,1], '--k')
    plt.plot(t, u_app[:,1], 'b')
    plt.axhline(robot.w_max, linestyle=':')
    plt.axhline(robot.w_min, linestyle=':')
    plt.title("omega")

    plt.suptitle(title)
    plt.show()


# ================= RUN DEMOS =================

# ---- Demo 1: No CBF ----
robot1 = MobileRobot()
res1 = simulate(robot1, use_cbf=False)

# ---- Demo 2: Fixed CBF ----
robot2 = MobileRobot()
res2 = simulate(robot2, use_cbf=True)

# ---- Demo 3: Restricted inputs ----
robot3 = MobileRobot(v_min=0.0, v_max=0.3, w_min=-0.3, w_max=0.3)
res3 = simulate(robot3, use_cbf=True)


# ================= PLOTS =================
plot_trajectory(res1, "Demo 1: No CBF (Crash)")
plot_trajectory(res2, "Demo 2: Fixed CBF (Safe)")
plot_trajectory(res3, "Demo 3: Restricted Inputs")

plot_barrier(res1, "Barrier (No CBF)")
plot_barrier(res2, "Barrier (CBF)")
plot_barrier(res3, "Barrier (Restricted)")

plot_controls(res1, robot1, "Controls (No CBF)")
plot_controls(res2, robot2, "Controls (CBF)")
plot_controls(res3, robot3, "Controls (Restricted)")


# ================= SUMMARY =================
print("\n===== SUMMARY =====")
print("No CBF:          min h =", res1["min_h"], " unsafe steps =", res1["unsafe"])
print("Fixed CBF:       min h =", res2["min_h"], " unsafe steps =", res2["unsafe"])
print("Restricted CBF:  min h =", res3["min_h"], " unsafe steps =", res3["unsafe"])