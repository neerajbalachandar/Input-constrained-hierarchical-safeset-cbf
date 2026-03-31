import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


class PointRobot2D:
    def __init__(self, ux_min=-1.0, ux_max=1.0, uy_min=-1.0, uy_max=1.0):
        self.ux_min = ux_min
        self.ux_max = ux_max
        self.uy_min = uy_min
        self.uy_max = uy_max
        self.u_bounds = [(ux_min, ux_max), (uy_min, uy_max)]


OBSTACLE_X, OBSTACLE_Y, OBSTACLE_R = 5.0, 0.0, 1.0
ALPHA = 2.0
GOAL = np.array([10.0, 0.0])


def barrier_function(x):
    px, py = x
    h = (px - OBSTACLE_X) ** 2 + (py - OBSTACLE_Y) ** 2 - OBSTACLE_R**2
    grad_h = np.array([2.0 * (px - OBSTACLE_X), 2.0 * (py - OBSTACLE_Y)])
    return h, grad_h


def nominal_controller(x):
    # Goal-seeking term (intentionally aggressive).
    u_goal = 1.8 * (GOAL - x)

    # Small circulation term near obstacle to break symmetry and encourage
    # passing around the obstacle instead of stalling on the boundary.
    rel = x - np.array([OBSTACLE_X, OBSTACLE_Y])
    dist = np.linalg.norm(rel)
    if dist < 1e-8:
        tangent = np.array([0.0, -1.0])
    else:
        tangent = np.array([-rel[1], rel[0]]) / dist

    influence = np.exp(-((dist - (OBSTACLE_R + 0.8)) ** 2) / (2.0 * 0.9**2))
    u_circ = 0.9 * influence * tangent
    return u_goal + u_circ


def safety_filter(robot, x, u_nominal):
    h, grad_h = barrier_function(x)

    # CBF condition for x_dot = u:
    # grad_h(x)^T u + alpha * h(x) >= 0
    rhs = -ALPHA * h

    u_nominal = np.array(
        [
            np.clip(u_nominal[0], robot.ux_min, robot.ux_max),
            np.clip(u_nominal[1], robot.uy_min, robot.uy_max),
        ]
    )

    # Fast path: if nominal already satisfies the CBF, keep it.
    if grad_h @ u_nominal >= rhs:
        return u_nominal

    def objective(u):
        return np.sum((u - u_nominal) ** 2)

    constraints = ({"type": "ineq", "fun": lambda u: grad_h @ u - rhs},)

    result = minimize(
        objective,
        x0=u_nominal,
        bounds=robot.u_bounds,
        constraints=constraints,
    )

    return result.x if result.success else np.array([0.0, 0.0])


def simulate(robot, use_cbf=True, dt=0.05, steps=320):
    x = np.array([0.0, 0.0])

    traj = [x.copy()]
    h_hist = []
    u_nom_hist = []
    u_app_hist = []

    min_h = np.inf
    unsafe_steps = 0
    cbf_active_steps = 0

    for _ in range(steps):
        u_nom = nominal_controller(x)
        u_nom = np.array(
            [
                np.clip(u_nom[0], robot.ux_min, robot.ux_max),
                np.clip(u_nom[1], robot.uy_min, robot.uy_max),
            ]
        )
        u_app = safety_filter(robot, x, u_nom) if use_cbf else u_nom
        if use_cbf and np.linalg.norm(u_app - u_nom) > 1e-8:
            cbf_active_steps += 1

        h, _ = barrier_function(x)
        min_h = min(min_h, h)
        if h < 0.0:
            unsafe_steps += 1

        x = x + u_app * dt

        traj.append(x.copy())
        h_hist.append(h)
        u_nom_hist.append(u_nom)
        u_app_hist.append(u_app)

    return {
        "traj": np.array(traj),
        "h": np.array(h_hist),
        "u_nom": np.array(u_nom_hist),
        "u_app": np.array(u_app_hist),
        "time": np.arange(0.0, steps * dt, dt),
        "min_h": min_h,
        "unsafe": unsafe_steps,
        "cbf_active_steps": cbf_active_steps,
        "goal_error": float(np.linalg.norm(x - GOAL)),
    }


def plot_trajectories(no_cbf, with_cbf):
    theta = np.linspace(0, 2 * np.pi, 300)
    obs_x = OBSTACLE_X + OBSTACLE_R * np.cos(theta)
    obs_y = OBSTACLE_Y + OBSTACLE_R * np.sin(theta)

    plt.figure(figsize=(8, 6))
    plt.fill(obs_x, obs_y, color="tomato", alpha=0.35, label="Obstacle")
    plt.plot(no_cbf["traj"][:, 0], no_cbf["traj"][:, 1], "k--", lw=2.0, label="No CBF")
    plt.plot(with_cbf["traj"][:, 0], with_cbf["traj"][:, 1], "b", lw=2.5, label="With CBF")
    plt.scatter(no_cbf["traj"][0, 0], no_cbf["traj"][0, 1], c="g", s=55, label="Start")
    plt.scatter(no_cbf["traj"][-1, 0], no_cbf["traj"][-1, 1], c="k", s=45, label="End (No CBF)")
    plt.scatter(with_cbf["traj"][-1, 0], with_cbf["traj"][-1, 1], c="b", s=45, label="End (With CBF)")
    plt.scatter(GOAL[0], GOAL[1], c="m", s=55, label="Goal")
    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("2D Trajectory: Obstacle Avoidance with CBF")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_barrier(no_cbf, with_cbf):
    low = min(np.min(no_cbf["h"]), np.min(with_cbf["h"]), -0.1)
    high = max(np.max(no_cbf["h"]), np.max(with_cbf["h"]), 0.1)

    plt.figure(figsize=(8, 4))
    plt.axhspan(low - 0.2, 0.0, color="tomato", alpha=0.15, label="Unsafe region h<0")
    plt.plot(no_cbf["time"], no_cbf["h"], "k--", lw=2.0, label="No CBF")
    plt.plot(with_cbf["time"], with_cbf["h"], "b", lw=2.0, label="With CBF")
    plt.axhline(0.0, color="r", linestyle=":", lw=1.8, label="Safety boundary h=0")
    plt.ylim(low - 0.2, high + 0.2)
    plt.xlabel("Time [s]")
    plt.ylabel("h(x)")
    plt.title("Barrier Function Over Time")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    robot = PointRobot2D(ux_min=-2.0, ux_max=2.0, uy_min=-2.0, uy_max=2.0)

    res_no_cbf = simulate(robot, use_cbf=False)
    res_with_cbf = simulate(robot, use_cbf=True)

    plot_trajectories(res_no_cbf, res_with_cbf)
    plot_barrier(res_no_cbf, res_with_cbf)

    print("\n===== SUMMARY =====")
    print(f"No CBF:   min h = {res_no_cbf['min_h']:.4f}, unsafe steps = {res_no_cbf['unsafe']}")
    print(f"With CBF: min h = {res_with_cbf['min_h']:.4f}, unsafe steps = {res_with_cbf['unsafe']}")
    print(f"CBF active steps = {res_with_cbf['cbf_active_steps']}")
    print(f"Goal error (No CBF)   = {res_no_cbf['goal_error']:.4f}")
    print(f"Goal error (With CBF) = {res_with_cbf['goal_error']:.4f}")


if __name__ == "__main__":
    main()
