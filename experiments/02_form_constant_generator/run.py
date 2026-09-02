"""
Experiment 02, Stage A: Ermentrout-Cowan scalar neural field on a periodic
cortical sheet, driven through the Turing bifurcation by mu, rendered to the
visual field through the inverse complex-log map.

    tau da/dt = -a + f( mu * (w * a) )

Parameters come from params.json (pinned by the 2026-09-03 amendment). Outputs:

    results-local/fields/{regime}_mu{mu}_s{seed}.npy   steady cortical field
    results-local/renders/{regime}_mu{mu}_s{seed}.png  visual-field render (512 x 512)
    results-local/metadata.json               every parameter, mu_c, k_c, per-run
                                              planform label and predicted class

Usage:  python run.py            # full sweep (2 regimes x 8 mu x 10 seeds = 160 runs, CPU, ~8 min)
        python run.py --quick    # 2 seeds, for a smoke test
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "params.json").read_text())
OUT = HERE / "results-local"


# ------------------------------------------------------------------ kernel ----
def w_hat(k2: np.ndarray, se: float, si: float) -> np.ndarray:
    """Fourier transform of unit-mass DoG: exp(-se^2 k^2/2) - exp(-si^2 k^2/2)."""
    return np.exp(-se**2 * k2 / 2) - np.exp(-si**2 * k2 / 2)


def critical_wavenumber(se: float, si: float) -> float:
    return float(np.sqrt(2 * np.log(si**2 / se**2) / (si**2 - se**2)))


def sigmoid(u, beta, theta):
    return 1.0 / (1.0 + np.exp(-beta * (u - theta)))


def f(u, beta, theta):
    return sigmoid(u, beta, theta) - sigmoid(0.0, beta, theta)


def f_prime0(beta, theta):
    s0 = sigmoid(0.0, beta, theta)
    return beta * s0 * (1 - s0)


# --------------------------------------------------------------- simulate ----
def simulate(mu: float, seed: int, n: int, L: float, se, si, beta, theta, dt, T, tau, init_amp, oriented_seed: bool):
    rng = np.random.default_rng(seed)
    a = init_amp * rng.standard_normal((n, n))
    # oriented seed of the same amplitude as the noise, random orientation psi per
    # seed: selects one member of the E(2)-symmetric roll family (Amendment 2)
    if oriented_seed:
        psi = rng.uniform(0, np.pi)
        kc = critical_wavenumber(se, si)
        xs = np.arange(n) * L / n
        X, Y = np.meshgrid(xs, xs, indexing="xy")
        a = a + init_amp * np.cos(kc * (np.cos(psi) * X + np.sin(psi) * Y))
    k = 2 * np.pi * np.fft.fftfreq(n, d=L / n)
    kx, ky = np.meshgrid(k, k, indexing="xy")
    W = w_hat(kx**2 + ky**2, se, si)
    steps = int(round(T / dt))
    for _ in range(steps):
        conv = np.real(np.fft.ifft2(W * np.fft.fft2(a)))
        a = a + (dt / tau) * (-a + f(mu * conv, beta, theta))
    return a


# ------------------------------------------------------------ planform label --
def label_planform(a: np.ndarray, L: float) -> dict:
    """Fixed rule from the amendment: blank / stripes(angle) / lattice."""
    std = float(a.std())
    if std < P["blank_std_threshold"]:
        return {"planform": "blank", "predicted_class": None, "std": std, "n_clusters": 0, "psi_deg": None}
    n = a.shape[0]
    F = np.abs(np.fft.fft2(a - a.mean())) ** 2
    F[0, 0] = 0.0
    k = 2 * np.pi * np.fft.fftfreq(n, d=L / n)
    kx, ky = np.meshgrid(k, k, indexing="xy")
    mask = F >= P["peak_power_fraction"] * F.max()
    ang = np.mod(np.degrees(np.arctan2(ky[mask], kx[mask])), 180.0)
    pw = F[mask]
    # cluster angles mod 180 with a 15 degree tolerance, strongest first
    clusters: list[float] = []
    for t in ang[np.argsort(pw)[::-1]]:
        if all(min(abs(t - c), 180 - abs(t - c)) > 15.0 for c in clusters):
            clusters.append(float(t))
    if len(clusters) >= 2:
        # hexagonal lattice = the three STRONGEST wavevector directions ~60 degrees
        # apart (a domain wall adds a second, weaker rotated triple); anything
        # else with several directions is a defect-ridden labyrinth
        cs = sorted(clusters[:3])
        gaps = [((cs[(i + 1) % len(cs)] - cs[i]) % 180.0) for i in range(len(cs))]
        is_hex = len(cs) == 3 and all(abs(g - 60.0) <= 15.0 for g in gaps)
        if is_hex:
            return {"planform": "lattice", "predicted_class": "lattice", "std": std,
                    "n_clusters": 3, "psi_deg": None}
        return {"planform": "labyrinth", "predicted_class": None, "std": std,
                "n_clusters": len(cs), "psi_deg": None}
    # stripes: psi is the angle of the wavevector to the x (ln r) axis, folded to [0, 90]
    psi = clusters[0]
    psi = min(psi, 180.0 - psi)
    s = P["stripe_angle_deg"]
    if psi <= s:
        cls, sub = "tunnel", "concentric"
    elif psi >= 90.0 - s:
        cls, sub = "tunnel", "radial"
    else:
        cls, sub = "spiral", "spiral"
    return {"planform": f"stripes_{sub}", "predicted_class": cls, "std": std,
            "n_clusters": 1, "psi_deg": psi}


# ----------------------------------------------------------------- render ----
def render(a: np.ndarray, L: float, px: int, r0: float, rmax: float, scale: float) -> Image.Image:
    n = a.shape[0]
    yy, xx = np.mgrid[0:px, 0:px]
    u = (xx - px / 2 + 0.5) / (px / 2)
    v = (yy - px / 2 + 0.5) / (px / 2)
    r = np.sqrt(u**2 + v**2)
    th = np.mod(np.arctan2(v, u), 2 * np.pi)
    x = L * (np.log(np.clip(r, r0, None)) - np.log(r0)) / (np.log(rmax) - np.log(r0))
    y = L * th / (2 * np.pi)
    vals = map_coordinates(a, [y / L * n, x / L * n], order=1, mode="wrap")
    img = 128.0 + 127.0 * np.clip(vals / scale, -1, 1)
    img[(r < r0) | (r > rmax)] = 128.0
    return Image.fromarray(img.astype(np.uint8), mode="L")


# ------------------------------------------------------------------- main ----
def main(quick: bool) -> None:
    n = P["grid_n"]; se, si = P["sigma_exc"], P["sigma_inh"]
    kc = critical_wavenumber(se, si)
    lam = 2 * np.pi / kc
    L = P["wavelengths_per_side"] * lam
    seeds = P["seeds"][:2] if quick else P["seeds"]
    mus = P["mu_grid_in_units_of_mu_c"] if not quick else [0.9, 1.5]
    (OUT / "fields").mkdir(parents=True, exist_ok=True)
    (OUT / "renders").mkdir(parents=True, exist_ok=True)
    runs, derived = [], {"k_c": kc, "lambda_c": lam, "L": L, "regimes": {}}
    t0 = time.time()
    for reg, rp in P["regimes"].items():
        beta, theta = rp["beta"], rp["theta"]
        mu_c = 1.0 / (f_prime0(beta, theta) * float(w_hat(np.array(kc**2), se, si)))
        derived["regimes"][reg] = {"mu_c": mu_c, "f_prime0": f_prime0(beta, theta)}
        for m in mus:
            for s in seeds:
                mu = m * mu_c
                a = simulate(mu, s, n, L, se, si, beta, theta, P["dt"], P["T"], P["tau"], P["init_amp"], rp["oriented_seed"])
                lab = label_planform(a, L)
                name = f"{reg}_mu{m:.2f}_s{s}"
                np.save(OUT / "fields" / f"{name}.npy", a.astype(np.float32))
                render(a, L, P["render_px"], P["r0"], P["r_max"], P["intensity_scale"]).save(OUT / "renders" / f"{name}.png")
                runs.append({"name": name, "regime": reg, "mu_over_mu_c": m, "mu": mu, "seed": s, **lab})
                print(f"[run] {name}  std={lab['std']:.3f}  {lab['planform']}  -> {lab['predicted_class']}")
    meta = {
        "experiment": "02_form_constant_generator", "stage": "A_scalar_field",
        "date": time.strftime("%Y-%m-%d"), "host": platform.platform(),
        "numpy": np.__version__, "params": P, "derived": derived,
        "runtime_s": round(time.time() - t0, 1), "runs": runs,
    }
    (OUT / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"[run] k_c={kc:.4f} L={L:.2f} mu_c={ {r: round(v['mu_c'], 3) for r, v in derived['regimes'].items()} }  {len(runs)} runs in {meta['runtime_s']}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
