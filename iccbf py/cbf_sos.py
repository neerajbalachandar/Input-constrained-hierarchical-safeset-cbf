import numpy as np
import sympy as sp
import cvxpy as cp
from sympy.polys.monomials import itermonomials


def _monomial_basis(vars_, degree):
    mons = list(itermonomials(vars_, degree))
    mons = sorted(mons, key=lambda m: (sp.total_degree(m), str(m)))
    return mons


def _state_box_from_cx(cx, x_syms):
    bounds = []
    for xi in x_syms:
        bi = 1.0
        for c in cx:
            if sp.expand(c) == sp.expand(xi**2 - 1):
                bi = 1.0
                break
        bounds.append((-bi, bi))
    return bounds


def _grid_samples(bounds, n_per_dim=9):
    axes = [np.linspace(lo, hi, n_per_dim) for lo, hi in bounds]
    mesh = np.meshgrid(*axes)
    return np.stack([m.ravel() for m in mesh], axis=1)


def _solve_problem(prob, solver, verbose):
    # Explicitly avoid MOSEK path; use open-source solvers only.
    solver_chain = []
    if solver is not None and str(solver).upper() not in ("MOSEK",):
        solver_chain.append(str(solver).upper())
    solver_chain += ["CLARABEL", "SCS"]

    last_err = None
    for s in solver_chain:
        try:
            prob.solve(solver=s, verbose=verbose)
            return
        except cp.SolverError as err:
            last_err = err
            continue
    if last_err is not None:
        raise last_err


def cbf_sos(plant, cbf_config, solver="CLARABEL", verbose=False):
    """
    Convex sampled analogue of MATLAB/YALMIP cbf_sos.m:
      - polynomial X(x), Y(x) with chosen bases
      - LMIs matching exp1..exp5 structure (sampled in x)
      - objective: maximize volume proxy of X0
    """

    n = int(plant["n"])
    nu = int(plant["nu"])
    nzx = int(plant["nzx"])

    A_sym = sp.Matrix(plant["A"])
    B_num = np.array(plant["B"], dtype=float)
    M_sym = sp.Matrix(plant["M"])
    x_sym = sp.Matrix(plant["x"])
    zx_sym = sp.Matrix(plant["zx"])
    zx_fcn = plant["zx_fcn"]

    P0 = np.array(plant["P0"], dtype=float)
    Du = np.array(plant["Du"], dtype=float)
    cx = list(plant["cx"])

    # Cx is symbolic in MATLAB main.m (depends on x)
    Cx_sym = sp.Matrix(plant.get("Cx", sp.zeros(0, nzx)))

    deg_X = int(cbf_config["deg_X"])
    deg_Y = int(cbf_config["deg_Y"])
    X_state_index = [int(i) for i in cbf_config["X_state_index"]]
    Y_state_index = [int(i) for i in cbf_config["Y_state_index"]]
    alpha = float(cbf_config.get("alpha", 1.0))
    include_input_limits = bool(cbf_config.get("include_input_limits", True))

    x_vars = list(x_sym)
    xX = [x_vars[i] for i in X_state_index]
    xY = [x_vars[i] for i in Y_state_index]
    polyX = _monomial_basis(xX, deg_X)
    polyY = _monomial_basis(xY, deg_Y)

    # lambdify evaluators
    x_tuple = tuple(x_vars)
    A_fcn = sp.lambdify(x_tuple, A_sym, "numpy")
    M_fcn = sp.lambdify(x_tuple, M_sym, "numpy")
    zx_eval = sp.lambdify(x_tuple, zx_sym, "numpy")
    Cx_fcn = sp.lambdify(x_tuple, Cx_sym, "numpy") if Cx_sym.rows > 0 else None
    fx_sym = A_sym * zx_sym
    fx_fcn = sp.lambdify(x_tuple, fx_sym, "numpy")

    # basis evaluators
    polyX_fcn = [sp.lambdify(tuple(xX), p, "numpy") for p in polyX]
    polyY_fcn = [sp.lambdify(tuple(xY), p, "numpy") for p in polyY]
    dpolyX = [[sp.diff(p, xx) for xx in xX] for p in polyX]
    dpolyX_fcn = [[sp.lambdify(tuple(xX), dp, "numpy") for dp in row] for row in dpolyX]

    # decision vars: X_coef(:,:,i), Y_coef(:,:,i), X0
    X_coef_vars = [cp.Variable((nzx, nzx), symmetric=True) for _ in range(len(polyX))]
    Y_coef_vars = [cp.Variable((nu, nzx)) for _ in range(len(polyY))]
    X0 = cp.Variable((nzx, nzx), symmetric=True)

    constraints = [X0 >> 1e-6 * np.eye(nzx)]

    bounds = _state_box_from_cx(cx, x_vars)
    samples = _grid_samples(bounds, n_per_dim=9)

    for xs in samples:
        xs = np.array(xs, dtype=float).reshape(-1)
        x_args = tuple(float(v) for v in xs)
        xX_args = tuple(float(xs[i]) for i in X_state_index)
        xY_args = tuple(float(xs[i]) for i in Y_state_index)

        # evaluate basis
        bx = np.array([float(f(*xX_args)) for f in polyX_fcn], dtype=float)
        by = np.array([float(f(*xY_args)) for f in polyY_fcn], dtype=float)

        # X(x), Y(x)
        Xx = 0
        for i in range(len(polyX)):
            Xx = Xx + X_coef_vars[i] * bx[i]

        Yx = 0
        for i in range(len(polyY)):
            Yx = Yx + Y_coef_vars[i] * by[i]

        # X_dot = sum_j dX/dx_j * fx(x)_idxj
        fx_val = np.array(fx_fcn(*x_args), dtype=float).reshape(n, 1)
        X_dot = 0
        for j, idx in enumerate(X_state_index):
            dX_j = 0
            for i in range(len(polyX)):
                dval = float(dpolyX_fcn[i][j](*xX_args))
                dX_j = dX_j + X_coef_vars[i] * dval
            X_dot = X_dot + dX_j * float(fx_val[idx, 0])

        A_val = np.array(A_fcn(*x_args), dtype=float)
        M_val = np.array(M_fcn(*x_args), dtype=float)

        # (1) CBF matrix condition (sampled x analogue of SOS exp1)
        tmp = M_val @ (A_val @ Xx + B_num @ Yx)
        constraints += [X_dot - (tmp + tmp.T) - alpha * Xx >> -1e-7 * np.eye(nzx)]

        # (2) input limits
        if include_input_limits:
            for j in range(Du.shape[0]):
                row = Du[j : j + 1, :] @ Yx
                psi = cp.bmat([[np.ones((1, 1)), row], [row.T, Xx]])
                constraints += [psi >> -1e-7 * np.eye(nzx + 1)]

        # (3) Xh subset condition via Phi_i
        if Cx_fcn is not None:
            Cx_val = np.array(Cx_fcn(*x_args), dtype=float)
            for i in range(Cx_val.shape[0]):
                row = Cx_val[i : i + 1, :] @ Xx
                phi = cp.bmat([[np.ones((1, 1)), row], [row.T, Xx]])
                constraints += [phi >> -1e-7 * np.eye(nzx + 1)]

        # (4) X >= X0
        constraints += [Xx - X0 >> -1e-7 * np.eye(nzx)]

        # (5) P0 >= X
        constraints += [P0 - Xx >> -1e-7 * np.eye(nzx)]

    # MATLAB uses geomean(X0). In CVXPY, log_det is standard concave volume proxy.
    objective = cp.Maximize(cp.log_det(X0 + 1e-8 * np.eye(nzx)))
    prob = cp.Problem(objective, constraints)
    _solve_problem(prob, solver=solver, verbose=verbose)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"cbf_sos failed with status: {prob.status}")

    X_coef_num = [np.array(v.value, dtype=float) for v in X_coef_vars]
    Y_coef_num = [np.array(v.value, dtype=float) for v in Y_coef_vars]

    def X_fcn(x_val):
        x_val = np.array(x_val, dtype=float).reshape(-1)
        xX_args = tuple(float(x_val[i]) for i in X_state_index)
        out = np.zeros((nzx, nzx), dtype=float)
        for i in range(len(polyX)):
            out += X_coef_num[i] * float(polyX_fcn[i](*xX_args))
        return out

    def Y_fcn(x_val):
        x_val = np.array(x_val, dtype=float).reshape(-1)
        xY_args = tuple(float(x_val[i]) for i in Y_state_index)
        out = np.zeros((nu, nzx), dtype=float)
        for i in range(len(polyY)):
            out += Y_coef_num[i] * float(polyY_fcn[i](*xY_args))
        return out

    def h_fcn(x_val):
        x_val = np.array(x_val, dtype=float).reshape(n, 1)
        z_val = np.array(zx_fcn(x_val), dtype=float).reshape(nzx, 1)
        X_val = X_fcn(x_val)
        X_val = 0.5 * (X_val + X_val.T)
        return float(1.0 - (z_val.T @ np.linalg.pinv(X_val) @ z_val)[0, 0])

    def h_fcn1(*args):
        return h_fcn(np.array(args, dtype=float).reshape(n, 1))

    return h_fcn, h_fcn1, X_coef_num, Y_coef_num, X_fcn, Y_fcn
