""" CODE BY : Shriram Hari
This code investigates the failure of obstacle avoidance for a 2nd order unicycle model using a heuristic Control Barrier Function.
This proves the need for non-hueristic based h(x) -> we look for parametrization and closed form solutions.
"""


import numpy as np
import casadi as ca
import matplotlib.pyplot as plt


class MobileRobotSim:
    def __init__(self, init_x, init_y, init_yaw, init_v=0.0, init_w=0.0):
        self.x = init_x
        self.y = init_y
        self.yaw = init_yaw
        self.v = init_v
        self.w = init_w

    def apply_dynamics(self, cmd_accel, cmd_yaw_accel, dt):
        self.v += cmd_accel * dt
        self.w += cmd_yaw_accel * dt
        
        self.x += self.v * np.cos(self.yaw) * dt
        self.y += self.v * np.sin(self.yaw) * dt
        self.yaw += self.w * dt
        
        self.yaw = np.arctan2(np.sin(self.yaw), np.cos(self.yaw))

def generate_route(start_pt, end_pt, resolution=150):
    base_x = np.linspace(start_pt[0], end_pt[0], resolution)
    base_y = np.linspace(start_pt[1], end_pt[1], resolution)
    
    curve_offset = 1.5 * np.sin(np.linspace(0, np.pi, resolution))
    base_y += curve_offset
    
    return base_x, base_y

def compute_nominal_commands(robot_state, tx, ty, dt):
    kp_v = 1.25
    kp_w = 2.2

    dx = tx - robot_state['x']
    dy = ty - robot_state['y']
    distance_error = np.hypot(dx, dy)
    target_yaw = np.arctan2(dy, dx)

    yaw_error = target_yaw - robot_state['yaw']
    yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))

    v_limit = 2.5                                                            
    w_limit = 2.0                                                      

    desired_v = np.clip(kp_v * distance_error, -v_limit, v_limit)
    desired_w = np.clip(kp_w * yaw_error, -w_limit, w_limit)

    req_a = (desired_v - robot_state['v']) / dt
    req_alpha = (desired_w - robot_state['w']) / dt

    return req_a, req_alpha

def evaluate_cbf(state, obs, l_shift):
    rx, ry, ryaw, rv, rw = state['x'], state['y'], state['yaw'], state['v'], state['w']
    ox, oy, ov, oyaw, orad = obs['x'], obs['y'], obs['v'], obs['yaw'], obs['r']

    p_rel = np.array([                                                   
        ox - rx - l_shift * np.cos(ryaw),
        oy - ry - l_shift * np.sin(ryaw)
    ])
    v_rel = np.array([
        ov * np.cos(oyaw) - rv * np.cos(ryaw) + l_shift * np.sin(ryaw) * rw,
        ov * np.sin(oyaw) - rv * np.sin(ryaw) - l_shift * np.cos(ryaw) * rw
    ])

    # naive heuristic h 
    safe_dist = orad + 0.4
    
    h_val = (np.linalg.norm(p_rel)**2 - safe_dist**2) + 0.8 * np.dot(p_rel, v_rel)
    
    return h_val

def solve_cbf_qp(nom_a, nom_alpha, state, obs, gamma, dt, l_shift):
    w_accel = 1.0
    w_steer = 1.5

    sx = ca.MX.sym('sx'); sy = ca.MX.sym('sy'); syaw = ca.MX.sym('syaw')
    sv = ca.MX.sym('sv'); sw = ca.MX.sym('sw')
    
    sox = ca.MX.sym('sox'); soy = ca.MX.sym('soy'); sov = ca.MX.sym('sov')
    soyaw = ca.MX.sym('soyaw'); sor = ca.MX.sym('sor')

    sys_params = ca.vertcat(sx, sy, syaw, sv, sw, sox, soy, sov, soyaw, sor)

    opt_a = ca.MX.sym('opt_a'); opt_alpha = ca.MX.sym('opt_alpha')
    controls = ca.vertcat(opt_a, opt_alpha)

    dp = ca.vertcat(sox - sx - l_shift * ca.cos(syaw),                       
                    soy - sy - l_shift * ca.sin(syaw))
    dv = ca.vertcat(
        sov * ca.cos(soyaw) - sv * ca.cos(syaw) + l_shift * ca.sin(syaw) * sw,
        sov * ca.sin(soyaw) - sv * ca.sin(syaw) - l_shift * ca.cos(syaw) * sw
    )


    safe_dist = sor + 0.4
    h_function = (ca.norm_2(dp)**2 - safe_dist**2) + 0.8 * ca.dot(dp, dv)

    dsx = sv * ca.cos(syaw); dsy = sv * ca.sin(syaw); dsyaw = sw
    dsv = opt_a; dsw = opt_alpha

    state_vars = ca.vertcat(sx, sy, syaw, sv, sw)                          
    state_dots = ca.vertcat(dsx, dsy, dsyaw, dsv, dsw)

    h_dot = ca.jtimes(h_function, state_vars, state_dots)                            

    cost = w_accel * (opt_a - nom_a)**2 + w_steer * (opt_alpha - nom_alpha)**2                  
    safety_constraint = [h_dot + gamma * h_function]

    solver_def = {'x': controls, 'f': cost, 'g': ca.vertcat(*safety_constraint), 'p': sys_params}    

    solver = ca.nlpsol('qp_solver', 'ipopt', solver_def, {                            
        'ipopt.print_level': 0, 'print_time': 0, 'ipopt.tol': 1e-4,
    })

    current_vals = [state['x'], state['y'], state['yaw'], state['v'], state['w'],              
                    obs['x'], obs['y'], obs['v'], obs['yaw'], obs['r']]


    phys_lbx = [-3.0, -2.0] # Max braking (-3 m/s^2), max steer right
    phys_ubx = [ 3.0,  2.0] # Max accel (+3 m/s^2), max steer left

    try:                                                                    
        solution = solver(x0=[nom_a, nom_alpha], 
                          lbx=phys_lbx, ubx=phys_ubx, 
                          lbg=0, ubg=ca.inf, p=current_vals)
        filtered_a = float(solution['x'][0])
        filtered_alpha = float(solution['x'][1])
    except RuntimeError:
        print("\n[CBF FAILED] QP Infeasible! Heuristic h asked for impossible inputs.")
        filtered_a = nom_a
        filtered_alpha = nom_alpha

    return filtered_a, filtered_alpha

def run_simulation():
   
    plt.rcParams.update({
        'font.size': 14,
        'font.family': 'serif',
        'axes.labelsize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.autolayout': True
    })

    dt = 0.05
    max_steps = 800
    cbf_gain = 3.0
    l_offset = 0.1
    
    pt_start = (-4.0, -4.0)
    pt_end = (6.0, 6.0)
    
    bot = MobileRobotSim(init_x=pt_start[0], init_y=pt_start[1], init_yaw=np.pi/4)                     
    hazard = {'x': 0.5, 'y': 1.0, 'v': 0.0, 'yaw': 0.0, 'r': 1.0}

    path_x, path_y = generate_route(pt_start, pt_end)

    log_x, log_y, log_time = [], [], []
    log_v, log_w, log_a, log_alpha = [], [], [], []
    log_h_val, log_h_time = [], []

    plt.ion()
    fig_traj, ax_traj = plt.subplots(figsize=(8, 6))
    ax_traj.set_xlim(-5, 7)
    ax_traj.set_ylim(-5, 7)
    ax_traj.set_xlabel('X Position [m]')
    ax_traj.set_ylabel('Y Position [m]')

    plot_bot, = ax_traj.plot([], [], 'bs', markersize=8, label='Robot')                         
    plot_trail, = ax_traj.plot([], [], 'b-', linewidth=2, alpha=0.6, label='Actual Trajectory')         
    plot_ref, = ax_traj.plot(path_x, path_y, 'g--', linewidth=2, label='Reference Curve')
    
    hazard_patch = plt.Circle((hazard['x'], hazard['y']), hazard['r'], color='red', alpha=0.4, label='Obstacle')
    ax_traj.add_patch(hazard_patch)
    ax_traj.legend(loc='upper left')

    for step in range(max_steps):
        current_time = step * dt
        target_idx = min(int(step * 0.75), len(path_x) - 1) 
        tgt_x = path_x[target_idx]
        tgt_y = path_y[target_idx]

        curr_state = {'x': bot.x, 'y': bot.y, 'yaw': bot.yaw, 'v': bot.v, 'w': bot.w}
        raw_a, raw_alpha = compute_nominal_commands(curr_state, tgt_x, tgt_y, dt)

        dist_to_hazard = np.hypot(bot.x - hazard['x'], bot.y - hazard['y'])
        if dist_to_hazard < (hazard['r'] + 2.0):
            cmd_a, cmd_alpha = solve_cbf_qp(raw_a, raw_alpha, curr_state, hazard, cbf_gain, dt, l_offset)
            h_metric = evaluate_cbf(curr_state, hazard, l_offset)                    
            log_h_val.append(h_metric)
            log_h_time.append(current_time)
        else:
            cmd_a, cmd_alpha = raw_a, raw_alpha

        bot.apply_dynamics(cmd_a, cmd_alpha, dt)                          

        log_x.append(bot.x); log_y.append(bot.y); log_time.append(current_time)
        log_v.append(bot.v); log_w.append(bot.w)
        log_a.append(cmd_a); log_alpha.append(cmd_alpha)

        plot_bot.set_data([bot.x], [bot.y])                                    
        plot_trail.set_data(log_x, log_y)
        fig_traj.canvas.draw()
        fig_traj.canvas.flush_events()
        plt.pause(0.001)

        if np.hypot(bot.x - pt_end[0], bot.y - pt_end[1]) < 0.35:               
            print(f"Goal Reached at t = {current_time:.2f}s")
            break

    plt.ioff()
    fig_traj.savefig('sim_trajectory.png', dpi=300, bbox_inches='tight')                                    


    fig_v = plt.figure(figsize=(8, 6))
    plt.plot(log_time, log_v, 'm-', linewidth=2)
    plt.xlabel('Time [s]')
    plt.ylabel('Linear Velocity [m/s]')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('sim_velocity.png', dpi=300, bbox_inches='tight')

   
    fig_a = plt.figure(figsize=(8, 6))
    plt.plot(log_time, log_a, 'c-', linewidth=2)
    plt.xlabel('Time [s]')
    plt.ylabel('Linear Acceleration [m/s²]')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('sim_acceleration.png', dpi=300, bbox_inches='tight')

    fig_w = plt.figure(figsize=(8, 6))
    plt.plot(log_time, log_w, 'g-', linewidth=2)
    plt.xlabel('Time [s]')
    plt.ylabel('Yaw Rate [rad/s]')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('sim_yaw_rate.png', dpi=300, bbox_inches='tight')

    fig_h = plt.figure(figsize=(8, 6))
    plt.xlabel('Time [s]')
    plt.ylabel('CBF Value (h)')
    plt.grid(True, linestyle='--', alpha=0.7)
    if log_h_val:
        plt.plot(log_h_time, log_h_val, 'r-', linewidth=2)
    else:
        plt.text(0.5, 0.5, 'Safety Filter Not Triggered', ha='center', va='center', transform=fig_h.gca().transAxes, fontsize=16)
    plt.savefig('sim_cbf_value.png', dpi=300, bbox_inches='tight')


    plt.show()

if __name__ == '__main__':
    run_simulation()