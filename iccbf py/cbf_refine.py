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


def _grid_samples(bounds, n_per_dim=19):
    axes = [np.linspace(lo, hi, n_per_dim) for lo, hi in bounds]
    mesh = np.meshgrid(*axes)
    return np.stack([m.ravel() for m in mesh], axis=1)


def _solve_problem(prob, solver, verbose=False):
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


def cbf_refine(plant, cbf_config, hx, max_iter, tol, solver="CLARABEL"):
    """
    MATLAB-like alternating refinement:
      1) fixed h(x), optimize u(x) and epsilon
      2) fixed u(x), optimize h(x)=y(x)^T Qh y(x), maximize mu
    """

    x = sp.Matrix(plant["x"])
    n = int(plant["n"])
    nu = int(plant["nu"])
    fx = sp.Matrix(plant["fx"])
    B = np.array(plant["B"], dtype=float)
    Du0 = np.array(plant["Du0"], dtype=float)
    cx = list(plant["cx"])

    alpha = float(cbf_config.get("alpha", 1.0))
    deg_ux_refine = int(cbf_config["deg_ux_refine"])
    deg_yx_refine = int(cbf_config["deg_yx_4_hx_refine"])
    include_input_limits = bool(cbf_config.get("include_input_limits", True))

    x_vars = list(x)
    x_tuple = tuple(x_vars)

    fx_fcn = sp.lambdify(x_tuple, fx, "numpy")
    h_fcn = sp.lambdify(x_tuple, hx, "numpy")
    dh_fcn = sp.lambdify(x_tuple, sp.Matrix([sp.diff(hx, xi) for xi in x]).T, "numpy")

    bounds = _state_box_from_cx(cx, x_vars)
    samples = _grid_samples(bounds, n_per_dim=17)

    # basis for u(x)
    bu = _monomial_basis(x_vars, deg_ux_refine)
    bu_fcn = [sp.lambdify(x_tuple, m, "numpy") for m in bu]

    # basis for h(x)
    yh = _monomial_basis(x_vars, deg_yx_refine)
    ny = len(yh)
    yh_fcn = [sp.lambdify(x_tuple, m, "numpy") for m in yh]
    dyh_fcn = [
        [sp.lambdify(x_tuple, sp.diff(m, x_vars[j]), "numpy") for m in yh]
        for j in range(n)
    ]

    min_eig_Qhs = []
    ux_sym = sp.Matrix([sp.Integer(0) for _ in range(nu)])

    for _ in range(max_iter):
        # =====================================================
        # STEP 1: fixed h, solve u polynomial + epsilon
        # =====================================================
        Uc = cp.Variable((nu, len(bu)))
        eps = cp.Variable()
        cst = []

        for xs in samples:
            x_args = tuple(float(v) for v in xs)
            dh_val = np.array(dh_fcn(*x_args), dtype=float).reshape(1, n)
            f_val = np.array(fx_fcn(*x_args), dtype=float).reshape(n, 1)
            h_val = float(h_fcn(*x_args))

            bu_val = np.array([float(f(*x_args)) for f in bu_fcn], dtype=float).reshape(-1, 1)
            u_val = Uc @ bu_val

            # CBF condition sampled (analogue of SOS with multipliers)
            cst += [dh_val @ (f_val + B @ u_val) + alpha * h_val >= eps]

            if include_input_limits:
                for j in range(Du0.shape[0]):
                    dj = Du0[j : j + 1, :]
                    cst += [dj @ u_val <= 1.0]
                    cst += [-dj @ u_val <= 1.0]

        # mild regularization for numerical conditioning only
        prob_u = cp.Problem(cp.Maximize(eps - 1e-6 * cp.sum_squares(Uc)), cst)
        _solve_problem(prob_u, solver=solver, verbose=False)
        if prob_u.status not in ("optimal", "optimal_inaccurate"):
            break

        Uc_num = np.array(Uc.value, dtype=float)
        ux_sym = []
        for i in range(nu):
            ui = 0
            for k in range(len(bu)):
                ui += float(Uc_num[i, k]) * bu[k]
            ux_sym.append(sp.expand(ui))
        ux_sym = sp.Matrix(ux_sym)

        # =====================================================
        # STEP 2: fixed u, solve h via Qh
        # =====================================================
        Qh = cp.Variable((ny, ny), symmetric=True)
        mu = cp.Variable()
        cst_h = [Qh[0, 0] == 1.0, Qh - mu * np.eye(ny) >> 0]

        for xs in samples:
            x_args = tuple(float(v) for v in xs)

            f_val = np.array(fx_fcn(*x_args), dtype=float).reshape(n, 1)
            bu_val = np.array([float(f(*x_args)) for f in bu_fcn], dtype=float).reshape(-1, 1)
            u_val = Uc_num @ bu_val
            fcl = f_val + B @ u_val

            y_val = np.array([float(f(*x_args)) for f in yh_fcn], dtype=float).reshape(-1, 1)
            h_expr = cp.quad_form(y_val, Qh)

            # dh/dx * fcl = sum_j (2 dy/dx_j^T Qh y) fcl_j
            dhf = 0
            for j in range(n):
                dyj = np.array([float(f(*x_args)) for f in dyh_fcn[j]], dtype=float).reshape(-1, 1)
                dh_j = 2.0 * (dyj.T @ Qh @ y_val)
                dhf += dh_j * float(fcl[j, 0])

            cst_h += [dhf + alpha * h_expr >= -1e-7]

            # state constraint analogue:
            # if any cx_i(x) > 0 (unsafe), require h(x) <= 0
            cx_vals = [float(sp.N(c.subs({x_vars[k]: x_args[k] for k in range(n)}))) for c in cx]
            if any(v > 0 for v in cx_vals):
                cst_h += [h_expr <= 0.0]

            # NOTE:
            # In Step-2, u(x) is fixed numerically from Step-1. Therefore,
            # input-limit inequalities here are numeric checks, not convex
            # constraints in Qh/mu. Adding them as CVXPY constraints would
            # create invalid ndarray booleans. We keep Step-2 focused on
            # h(x) search, while input limits are enforced in Step-1.

        prob_h = cp.Problem(cp.Maximize(mu), cst_h)
        _solve_problem(prob_h, solver=solver, verbose=False)
        if prob_h.status not in ("optimal", "optimal_inaccurate"):
            break

        Qh_num = np.array(Qh.value, dtype=float)
        hx = sp.expand((sp.Matrix(yh).T * sp.Matrix(Qh_num) * sp.Matrix(yh))[0])
        dh_fcn = sp.lambdify(x_tuple, sp.Matrix([sp.diff(hx, xi) for xi in x]).T, "numpy")
        h_fcn = sp.lambdify(x_tuple, hx, "numpy")

        min_eig = float(np.min(np.linalg.eigvalsh(0.5 * (Qh_num + Qh_num.T))))
        min_eig_Qhs.append(min_eig)

        if len(min_eig_Qhs) >= 3:
            if (
                abs(min_eig_Qhs[-1] - min_eig_Qhs[-2]) < tol
                and abs(min_eig_Qhs[-2] - min_eig_Qhs[-3]) < tol
            ):
                break

    dh_dx = sp.Matrix([sp.diff(hx, xi) for xi in x]).T
    return hx, ux_sym, dh_dx, np.array(min_eig_Qhs, dtype=float)
