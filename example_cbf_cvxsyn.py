import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.io import savemat
from scipy.optimize import minimize

from cbf_sos import cbf_sos
from cbf_refine import cbf_refine


# ============================================================
# SETTINGS (aligned with reference main.m)
# ============================================================
save_file = False
print_file = True
linear_like_form = 2       # {1, 2}
include_input_limits = True
min_norm_control = False

deg_X = 0
deg_Y = 2
X_states_index = [1]       # MATLAB index 2
Y_states_index = [0, 1]    # MATLAB index [1 2]


# ============================================================
# DYNAMICS AND CONSTRAINTS
# ============================================================
n = 2
nu = 1

x1, x2 = sp.symbols("x1 x2", real=True)
x = sp.Matrix([x1, x2])
x_store = x

if linear_like_form == 1:
    zx = x
    zx_fcn = lambda xv: np.array(xv, dtype=float).reshape(2, 1)
    A = sp.Matrix([[1, 0], [-1, 0.5 + x2**2]])
    B = sp.Matrix([[1], [0]])
else:
    zx = sp.Matrix([x1, x2, x2**3])
    def zx_fcn(xv):
        xv = np.array(xv, dtype=float).reshape(2,)
        return np.array([[xv[0]], [xv[1]], [xv[1] ** 3]], dtype=float)
    A = sp.Matrix([[1, 0, 0], [-1, 0.5, 1]])
    B = sp.Matrix([[1], [0]])

nzx = len(zx)
fx = A * zx
M = zx.jacobian(x)

def f_fcn(xv):
    xv = np.array(xv, dtype=float).reshape(2,)
    return np.array([xv[0], -xv[0] + 0.5 * xv[1] + xv[1] ** 3], dtype=float).reshape(2, 1)

def dynamics(_t, xv, u):
    xv = np.array(xv, dtype=float).reshape(2,)
    return (f_fcn(xv) + np.array([[float(u)], [0.0]])).reshape(2,)

cx = [x1**2 - 1, x2**2 - 1]
c1_fcn = sp.lambdify((x1, x2), cx[0], "numpy")
c2_fcn = sp.lambdify((x1, x2), cx[1], "numpy")

if linear_like_form == 1:
    Cx = sp.Matrix([[x1, 0], [0, x2]])
else:
    Cx = sp.Matrix([[x1, 0, 0], [0, x2, 0]])

Du0 = sp.Matrix([[1]])   # Du0 * u <= 1
Du = sp.Matrix([[1]])    # |Du * u| < 1

if linear_like_form == 1:
    P0 = 2 * np.eye(nzx)
    h0_x = 1 - (zx.T * zx)[0] / P0[0, 0]
    h0_fcn = sp.lambdify((x1, x2), h0_x, "numpy")
else:
    P0 = 3 * np.eye(nzx)
    h0_x = 1 - (zx.T * zx)[0] / P0[0, 0]
    h0_fcn = sp.lambdify((x1, x2), h0_x, "numpy")


# ============================================================
# PLOT INITIAL CONSTRAINTS
# ============================================================
xx = np.linspace(-1.5, 1.5, 300)
yy = np.linspace(-1.5, 1.5, 300)
XX, YY = np.meshgrid(xx, yy)
H0 = h0_fcn(XX, YY)
C1 = c1_fcn(XX, YY)
C2 = c2_fcn(XX, YY)

plt.figure(1, figsize=(6.5, 6))
plt.contour(XX, YY, H0, levels=[0], colors="k", linewidths=1.8)
plt.contour(XX, YY, C1, levels=[0], colors="r", linewidths=1.5)
plt.contour(XX, YY, C2, levels=[0], colors="r", linewidths=1.5)
plt.gca().set_aspect("equal")
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.grid(alpha=0.2)


# ============================================================
# STEP 1: INITIAL CBF SYNTHESIS
# ============================================================
plant = {
    "n": n,
    "nu": nu,
    "nzx": nzx,
    "A": A,
    "B": B,
    "M": M,
    "fx": fx,
    "dynamics": dynamics,
    "x": x,
    "x_store": x_store,
    "zx": zx,
    "zx_fcn": zx_fcn,
    "P0": P0,
    "h0_x": h0_x,
    "h0_fcn": h0_fcn,
    "Du": Du,
    "Du0": Du0,
    "Cx": Cx,
    "cx": cx,
}

cbf_config = {
    "deg_X": deg_X,
    "deg_Y": deg_Y,
    "X_state_index": X_states_index,
    "Y_state_index": Y_states_index,
    "deg_Lcbf": 4,
    "deg_Lu": 2,
    "deg_Lx": 2,
    "deg_L_X0": 2,
    "deg_L_P0": 2,
    "alpha": 1.0,
    "include_input_limits": include_input_limits,
}

print("Running initial CBF synthesis (CLARABEL/SCS)...")
h_init_fcn, h_init_fcn1, X_coef, Y_coef, X_fcn, Y_fcn = cbf_sos(
    plant, cbf_config, solver="CLARABEL", verbose=False
)

Hinit = np.vectorize(h_init_fcn1)(XX, YY)
plt.contour(XX, YY, Hinit, levels=[0], colors="b", linestyles="--", linewidths=1.8)


# ============================================================
# STEP 2: ITERATIVE CBF REFINEMENT
# ============================================================
if len(X_coef) != 1:
    raise ValueError("X must be constant (deg_X=0) for polynomial h(x) refinement.")

P_refine = np.linalg.pinv(X_coef[0])   # MATLAB: eye(nzx)/X_coef(:,:,1)
hx = sp.expand(1 - (zx.T * sp.Matrix(P_refine) * zx)[0])

if linear_like_form == 1:
    deg_ux_refine = 3
    deg_yx_4_hx_refine = 1
else:
    deg_ux_refine = 5
    deg_yx_4_hx_refine = 3

cbf_config["deg_ux_refine"] = deg_ux_refine
cbf_config["deg_yx_4_hx_refine"] = deg_yx_4_hx_refine

print("Running iterative CBF refinement (CLARABEL/SCS)...")
hx_refined, ux, dh_dx, min_eig_Qhs = cbf_refine(
    plant=plant,
    cbf_config=cbf_config,
    hx=hx,
    max_iter=50,
    tol=1e-3,
    solver="CLARABEL",
)

h_fcn = sp.lambdify((x1, x2), hx_refined, "numpy")
u_fcn = sp.lambdify((x1, x2), ux[0], "numpy")
dh_dx_fcn = sp.lambdify((x1, x2), dh_dx, "numpy")
h_fcn1 = lambda a, b: float(h_fcn(a, b))

Href = h_fcn(XX, YY)
plt.contour(XX, YY, Href, levels=[0], colors="g", linewidths=2.0)
plt.xlim([-1.1, 1.1])
plt.ylim([-1.3, 1.3])


# ============================================================
# SIMULATION
# ============================================================
x0 = np.array([[-0.5], [-0.5]], dtype=float)
if linear_like_form == 2:
    x0 = np.array([[-0.8], [-0.7]], dtype=float)

dt = 0.005
duration = 5.0
t_vec = np.arange(0.0, duration + dt, dt)
T_steps = len(t_vec) - 1

xTraj = np.zeros((n, T_steps + 1))
uTraj = np.zeros((nu, T_steps))
x_now = x0.copy()
xTraj[:, [0]] = x_now

u_bound = 1.0 / np.diag(np.array(Du, dtype=float))

def min_norm_control_qp(phi0, phi1, ub):
    phi1 = np.array(phi1, dtype=float).reshape(1, nu)
    phi0 = float(phi0)

    def obj(u):
        return 0.5 * float(np.dot(u, u))

    cons = [{"type": "ineq", "fun": lambda u: float(phi1 @ u.reshape(-1, 1)) + phi0}]
    bnds = [(-ub, ub)] * nu
    res = minimize(obj, x0=np.zeros(nu), bounds=bnds, constraints=cons, method="SLSQP")
    if res.success:
        return np.array(res.x, dtype=float).reshape(nu, 1)
    return np.zeros((nu, 1))

for i in range(T_steps):
    x_i = x_now.reshape(-1,)
    if not min_norm_control:
        u = np.array([[float(u_fcn(float(x_i[0]), float(x_i[1])))]], dtype=float)
    else:
        dh_dx0 = np.array(dh_dx_fcn(float(x_i[0]), float(x_i[1])), dtype=float).reshape(1, n)
        phi0 = float(dh_dx0 @ f_fcn(x_i) + cbf_config["alpha"] * h_fcn(float(x_i[0]), float(x_i[1])))
        phi1 = dh_dx0 @ np.array(B, dtype=float)
        u = min_norm_control_qp(phi0, phi1, float(u_bound[0]))

    u = np.clip(u, -u_bound.reshape(-1, 1), u_bound.reshape(-1, 1))
    uTraj[:, [i]] = u

    sol = solve_ivp(
        fun=lambda tt, xx: dynamics(tt, xx, float(u[0, 0])),
        t_span=(t_vec[i], t_vec[i + 1]),
        y0=x_now.reshape(-1,),
        method="RK23",
    )
    x_now = sol.y[:, -1].reshape(n, 1)
    xTraj[:, [i + 1]] = x_now


# ============================================================
# PLOTS
# ============================================================
plt.figure(1)
plt.plot(xTraj[0, :], xTraj[1, :], "k-.", linewidth=1)
plt.scatter(xTraj[0, 0], xTraj[1, 0], c="k", s=25)
plt.legend(
    [r"$h^{\mathrm{init}}(x)=0$", r"$h(x)=0$"],
    loc="upper center",
    ncol=2,
)

plt.figure(2, figsize=(6, 3))
plt.plot(t_vec[:-1], uTraj[0, :], "k", linewidth=1)
plt.plot(t_vec, -1 * np.ones_like(t_vec), "r--", linewidth=1)
plt.plot(t_vec, 1 * np.ones_like(t_vec), "r--", linewidth=1)
plt.ylabel(r"$u$")
plt.xlabel("Time (s)")
plt.ylim([-1.1, 1.1])
plt.grid(alpha=0.2)


# ============================================================
# SAVE
# ============================================================
if save_file:
    file_name = (
        f"exp1_ulim_{int(include_input_limits)}_"
        f"linearlike_{linear_like_form}_"
        f"degY_{deg_Y}_"
        f"deg_ux_refine_{deg_ux_refine}.mat"
    )
    savemat(
        file_name,
        {
            "t_vec": t_vec,
            "xTraj": xTraj,
            "uTraj": uTraj,
            "min_eig_Qhs": np.array(min_eig_Qhs, dtype=float),
            "include_input_limits": np.array([[int(include_input_limits)]], dtype=float),
            "linear_like_form": np.array([[linear_like_form]], dtype=float),
            "deg_Y": np.array([[deg_Y]], dtype=float),
            "deg_ux_refine": np.array([[deg_ux_refine]], dtype=float),
            "hx_refined_str": np.array([str(hx_refined)], dtype=object),
            "ux_str": np.array([str(ux)], dtype=object),
        },
    )
    print(f"Saved: {file_name}")

if print_file:
    fig_name = f"exp1_ulim_{int(include_input_limits)}_linearlike_{linear_like_form}_u"
    plt.figure(2)
    plt.savefig(f"{fig_name}.pdf", dpi=150, bbox_inches="tight")

print("\n===== SUMMARY =====")
print(f"min h along trajectory = {np.min(h_fcn(xTraj[0, :], xTraj[1, :])):.4f}")
print(f"max |u| = {np.max(np.abs(uTraj)):.4f}")
if len(min_eig_Qhs) > 0:
    print(f"last min_eig_Qh = {min_eig_Qhs[-1]:.4f}")

plt.show()
