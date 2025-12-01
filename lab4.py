"""
Молькова Настя
lab4.py
Симулятор и оптимизация s-Q модели с задержкой для трёх задач:
 - эндокринология (глюкоза)
 - DevOps (масштабирование)
 - умный полив (теплица)

Запуск:
  python lab4.py

Зависимости:
  numpy, scipy, matplotlib
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time

RNG = np.random.default_rng(12345)

# ---------------------------
# Общий симулятор (дискретизация dt)
# ---------------------------
def simulate_once(params, s, Q, T=24.0, dt=0.1, verbose=False):

    model = params['model']
    X = params.get('X0', params.get('Xmax', 100.0))
    Xmax = params.get('Xmax', 100.0)
    obs_dt = params.get('observation_dt', 1.0)
    next_obs = 0.0
    t = 0.0
    order_pending = False
    order_delivery_time = np.inf

    orders = []
    ts = []
    xs = []

    shortage_integral = 0.0
    holding_integral = 0.0
    n_orders = 0

    # helper to sample lead time
    Lparam = params.get('lead_time', 0.0)
    def sample_lead():
        if callable(Lparam):
            return Lparam()
        else:
            return Lparam

    while t < T - 1e-9:

        # compute consumption rate
        if model == 'glucose':
            cp = params['consumption_params']
            base = cp.get('base', 0.05)
            if RNG.random() < cp.get('spike_prob_per_dt', 0.01):
                spike = cp.get('spike_amp', 0.5) * RNG.random()
            else:
                spike = 0.0
            d = base + spike

        elif model == 'devops':
            cp = params['consumption_params']
            base = cp.get('base', 1.0)
            if RNG.random() < cp.get('burst_prob_per_dt', 0.01):
                burst = cp.get('burst_amp', 5.0) * RNG.random()
            else:
                burst = 0.0
            d = base + burst

        elif model == 'irrigation':
            cp = params['consumption_params']
            trend = cp.get('trend', 0.8)
            noise = cp.get('noise_std', 0.1) * RNG.normal()
            d = max(trend + noise, 0.0)
        else:
            raise ValueError("Unknown model")

        # Euler step
        dX = -d * dt
        X = X + dX

        # Физически влажность < 0 невозможна → ограничиваем
        if model == 'irrigation':
            X = max(X, 0.0)

        # accumulate costs integrals
        if X >= 0:
            holding_integral += X * dt
        else:
            shortage_integral += (-X) * dt

        # check order delivery
        if order_pending and t >= order_delivery_time - 1e-12:
            X = min(X + Q, Xmax)
            order_pending = False
            orders.append(order_delivery_time)
            n_orders += 1

        # observation event
        if t >= next_obs - 1e-12:
            meas_noise = params.get('meas_noise_std', 0.0) * RNG.normal()
            X_tilde = X + meas_noise

            if (X_tilde <= s) and (not order_pending):
                Ls = sample_lead()
                order_pending = True
                order_delivery_time = t + Ls
            next_obs += obs_dt

        ts.append(t)
        xs.append(X)
        t += dt

    # compute averages
    avg_holding = holding_integral / T
    avg_shortage = shortage_integral / T

    Ch = params['costs']['Ch']
    Cs = params['costs']['Cs']
    Co = params['costs']['Co']

    J = Ch * avg_holding + Cs * avg_shortage + Co * (n_orders / T)

    return {
        'ts': np.array(ts),
        'xs': np.array(xs),
        'orders': orders,
        'J': J,
        'avg_holding': avg_holding,
        'avg_shortage': avg_shortage,
        'n_orders': n_orders
    }

# ---------------------------
# Функция оценки J(s,Q) с Monte-Carlo
# ---------------------------
def estimate_cost(params, s, Q, T=24.0, dt=0.1, mc_runs=40):
    costs = []
    for _ in range(mc_runs):
        r = simulate_once(params, s, Q, T=T, dt=dt)
        costs.append(r['J'])
    return float(np.mean(costs)), float(np.std(costs)/np.sqrt(len(costs)))

# ---------------------------
# Сетка и оптимизация
# ---------------------------
def grid_search(params, s_grid, Q_grid, mc_runs=40, T=24.0, dt=0.1):
    best = None
    results = {}
    for s in s_grid:
        for Q in Q_grid:
            meanJ, se = estimate_cost(params, s, Q, T=T, dt=dt, mc_runs=mc_runs)
            results[(s,Q)] = (meanJ, se)
            if (best is None) or (meanJ < best[0]):
                best = (meanJ, s, Q)
    return best, results

def local_optimize(params, s0, Q0, bounds, T=24.0, dt=0.1, mc_runs=30):

    def obj(x):
        s, Q = float(x[0]), float(x[1])
        meanJ, _ = estimate_cost(params, s, Q, T=T, dt=dt, mc_runs=mc_runs)
        return meanJ

    x0 = np.array([s0, Q0])

    res = minimize(
        obj, x0,
        method='Nelder-Mead',
        bounds=bounds,
        options={'maxiter': 100, 'adaptive': True}
    )

    return res

# ---------------------------
# Примеры параметров для задач
# ---------------------------
def default_params_glucose():
    return {
        'model': 'glucose',
        'X0': 5.5,
        'Xmax': 12.0,
        'observation_dt': 0.25,
        'lead_time': 0.5,
        'meas_noise_std': 0.05,
        'consumption_params': {'base': 0.02, 'spike_prob_per_dt': 0.005, 'spike_amp': 0.5},
        'costs': {'Ch': 0.1, 'Cs': 10.0, 'Co': 0.5}
    }

def default_params_devops():
    return {
        'model': 'devops',
        'X0': 10.0,
        'Xmax': 100.0,
        'observation_dt': 0.1,
        'lead_time': lambda: max(0.05, 0.5 * RNG.normal() + 0.5),
        'meas_noise_std': 0.0,
        'consumption_params': {'base': 1.0, 'burst_prob_per_dt': 0.01, 'burst_amp': 6.0},
        'costs': {'Ch': 0.05, 'Cs': 5.0, 'Co': 0.2}
    }

def default_params_irrigation():
    return {
        'model': 'irrigation',
        'X0': 40.0,
        'Xmax': 100.0,
        'observation_dt': 1.0,
        'lead_time': 0.5,
        'meas_noise_std': 0.2,     # было 0.5 — слишком шумно
        'consumption_params': {'trend': 0.8, 'noise_std': 0.1},
        'costs': {'Ch': 0.02, 'Cs': 1.0, 'Co': 0.1}
    }

# ---------------------------
# Пример запуска и графики
# ---------------------------
def run_example_for_model(model_name):

    if model_name == 'glucose':
        params = default_params_glucose()
        s_grid = np.linspace(3.0, 6.0, 7)
        Q_grid = np.linspace(0.5, 3.5, 7)

    elif model_name == 'devops':
        params = default_params_devops()
        s_grid = np.linspace(1, 15, 8)
        Q_grid = np.linspace(1, 25, 9)

    elif model_name == 'irrigation':
        params = default_params_irrigation()
        s_grid = np.linspace(0, 45, 10)   # раньше было (20,45)
        Q_grid = np.linspace(5, 40, 8)
    else:
        raise ValueError("Unknown model")

    print(f"=== Running grid search for {model_name} ===")
    start = time.time()

    best, grid_res = grid_search(params, s_grid, Q_grid, mc_runs=50, T=48.0, dt=0.1)

    print("Grid search finished in {:.1f}s".format(time.time() - start))
    print(f"Best grid result: J={best[0]:.4f}, s={best[1]}, Q={best[2]}")

    # ---- Local optimization with bounds ----
    bounds = [(0, params['Xmax']), (0.1, params['Xmax'])]
    print("Running local optimization...")
    res = local_optimize(params, best[1], best[2], bounds=bounds, T=48.0, dt=0.1, mc_runs=40)

    print("Local optimization result:", res)
    s_opt, Q_opt = res.x[0], res.x[1]

    # final simulation
    r = simulate_once(params, s_opt, Q_opt, T=72.0, dt=0.1)
    print("Final J:", r['J'], "orders:", r['n_orders'], "avg_hold:", r['avg_holding'], "avg_short:", r['avg_shortage'])

    plt.figure(figsize=(8, 3))
    plt.plot(r['ts'], r['xs'])
    plt.axhline(s_opt, color='red', linestyle='--', label=f"s_opt={s_opt:.2f}")
    plt.title(f"{model_name}: X(t) with s_opt={s_opt:.2f}, Q_opt={Q_opt:.2f}")
    plt.xlabel("time")
    plt.ylabel("X")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{model_name}_opt.png")


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    #run_example_for_model('irrigation')
    #run_example_for_model('devops')
    run_example_for_model('glucose')

