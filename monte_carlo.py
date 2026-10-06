"""Does the CFAR detector behave the way the theory says?

1. False alarms: run noise-only frames and count how often a cell crosses
   the threshold, for a few design values of P_fa.
2. Detection: put one target at 100 m, +5 m/s and see how often it is found
   as the SNR goes up.

Run:  python monte_carlo.py               (about 3 minutes; results in ./results,
                                          figures in ./figures/report)
      python monte_carlo.py --plot-only   (figures from the saved results)
"""

from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import signal, stats
import matplotlib.pyplot as plt

import fmcw_radar_sim as m
from fmcw_radar_sim import BLUE, ORANGE, INK2

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
FIG = ROOT / "figures" / "report"

N_NOISE_FRAMES = 600
N_TRIALS = 150
SNR_PER_SAMPLE_DB = np.arange(-35.0, -23.0, 1.0)
PFA_DESIGN = np.array([1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
PFA_DETECT = 1e-6
TARGET = m.Target(100.0, 5.0, 1.0, "MC")
MIN_RANGE_BIN = 3

N_TRAIN = 248
ALPHA = N_TRAIN * (PFA_DESIGN ** (-1 / N_TRAIN) - 1)

radar = m.Radar()
h = m.antialias_filter(radar)


def noise_frame(seed):
    rng = np.random.default_rng(seed)
    x = m.adc(radar, m.simulate_beat(radar, [], h, rng), h)
    power, _, _ = m.range_doppler(radar, x)
    _, _, noise = m.ca_cfar_2d(power)
    ratio = (power / noise)[MIN_RANGE_BIN:]
    counts = np.array([(ratio > a).sum() for a in ALPHA])
    return counts, ratio.size, power.mean(axis=1)


def iid_frame(seed):
    """Same CFAR on fully independent noise cells, as a reference."""
    power = np.random.default_rng(seed).exponential(size=(256, radar.n_chirps))
    _, _, noise = m.ca_cfar_2d(power)
    ratio = (power / noise)[MIN_RANGE_BIN:]
    return np.array([(ratio > a).sum() for a in ALPHA]), ratio.size


def detection_trial(args):
    snr_db, seed = args
    rng = np.random.default_rng(seed)
    r = m.Radar(snr_ref_db=snr_db)              # target is 1 m^2 at 100 m, so its SNR is snr_db
    x = m.adc(r, m.simulate_beat(r, [TARGET], h, rng), h)
    power, f_beat, f_dopp = m.range_doppler(r, x)
    mask, _, noise = m.ca_cfar_2d(power, pfa=PFA_DETECT)
    dets = m.extract_detections(r, power, mask, noise, f_beat, f_dopp)
    return any(abs(d.r - TARGET.r0) < 3 and abs(d.v - TARGET.v) < 2 for d in dets)


def pd_ca_cfar(snr_lin, pfa, n, rng=np.random.default_rng(0), n_mc=20000):
    """Theoretical Pd for a steady target, averaged over the random noise estimate."""
    alpha = n * (pfa ** (-1 / n) - 1)
    est = rng.gamma(n, 1 / n, n_mc)
    return np.array([stats.ncx2.sf(2 * alpha * est, 2, 2 * s).mean() for s in np.atleast_1d(snr_lin)])


def pd_known_noise(snr_lin, pfa):
    return stats.ncx2.sf(-2 * np.log(pfa), 2, 2 * np.atleast_1d(snr_lin))


def main():
    RES.mkdir(exist_ok=True)

    with Pool() as pool:
        out = pool.map(noise_frame, range(10_000, 10_000 + N_NOISE_FRAMES))
        counts = sum(o[0] for o in out)
        cells = sum(o[1] for o in out)
        noise_profile = np.mean([o[2] for o in out], axis=0)
        pfa_meas = counts / cells
        ref = pool.map(iid_frame, range(N_NOISE_FRAMES))
        pfa_iid = sum(o[0] for o in ref) / sum(o[1] for o in ref)

        jobs = [(s, 20_000 + 1000 * i + k) for i, s in enumerate(SNR_PER_SAMPLE_DB) for k in range(N_TRIALS)]
        hits = np.array(pool.map(detection_trial, jobs)).reshape(len(SNR_PER_SAMPLE_DB), N_TRIALS)
    pd_meas = hits.mean(axis=1)

    # how far the two FFTs lift the target above the noise in its range bin
    n_s = int(round(radar.t_adc * radar.fs_adc))
    k_target = int(round(2 * radar.slope * TARGET.r0 / m.C / (radar.fs_adc / 512)))
    gain = (signal.windows.hann(n_s).sum() * signal.windows.hann(radar.n_chirps).sum()) ** 2
    gain_db = 10 * np.log10(gain / noise_profile[k_target])
    snr_map_db = SNR_PER_SAMPLE_DB + gain_db

    np.savetxt(RES / "pfa.csv", np.column_stack([PFA_DESIGN, pfa_meas, counts, pfa_iid]),
               delimiter=",", header="pfa_design,pfa_measured,false_alarms,pfa_iid_reference", comments="", fmt="%.6g")
    np.savetxt(RES / "pd.csv", np.column_stack([SNR_PER_SAMPLE_DB, snr_map_db, pd_meas]),
               delimiter=",", header="snr_per_sample_db,snr_map_db,pd_measured", comments="", fmt="%.4f")

    print(f"{N_NOISE_FRAMES} noise frames, {cells:.3g} cells")
    for p, q, c, r in zip(PFA_DESIGN, pfa_meas, counts, pfa_iid):
        print(f"  design P_fa {p:.0e}   measured {q:.2e}   ({c} false alarms)   i.i.d. reference {r:.2e}")
    print(f"Processing gain to the range-Doppler cell: {gain_db:.1f} dB")
    for s, sm, p in zip(SNR_PER_SAMPLE_DB, snr_map_db, pd_meas):
        print(f"  SNR {s:6.1f} dB/sample  {sm:5.1f} dB in cell   P_d = {p:.2f}")
    plot()


def effective_n(pfa_meas):
    """How many truly independent training cells would explain the measured P_fa."""
    grid = np.arange(10, N_TRAIN + 1)
    err = [np.sum((np.log10((1 + ALPHA / n) ** (-n)) - np.log10(pfa_meas)) ** 2) for n in grid]
    return int(grid[np.argmin(err)])


def plot():
    FIG.mkdir(parents=True, exist_ok=True)
    pfa = np.loadtxt(RES / "pfa.csv", delimiter=",", skiprows=1)
    pd = np.loadtxt(RES / "pd.csv", delimiter=",", skiprows=1)
    pfa_meas, snr_map_db, pd_meas = pfa[:, 1], pd[:, 1], pd[:, 2]
    n_eff = effective_n(pfa_meas)
    print(f"Effective number of independent training cells: {n_eff}")

    plt.rcParams.update({"font.size": 8, "legend.fontsize": 7.5, "xtick.labelsize": 7.5,
                         "ytick.labelsize": 7.5, "figure.facecolor": "#ffffff",
                         "axes.facecolor": "#ffffff", "savefig.facecolor": "#ffffff"})

    p_fine = np.logspace(-7, np.log10(2e-2), 100)
    a_fine = N_TRAIN * (p_fine ** (-1 / N_TRAIN) - 1)
    fig, ax = plt.subplots(figsize=(2.9, 2.5))
    ax.loglog(p_fine, p_fine, color=INK2, lw=0.8, ls="--", label="Design")
    ax.loglog(p_fine, (1 + a_fine / n_eff) ** (-n_eff), color=ORANGE, lw=1.2,
              label=rf"Model, $N_{{\mathrm{{eff}}}}$ = {n_eff}")
    ax.loglog(PFA_DESIGN, pfa_meas, "o", color=BLUE, ms=5, label="Simulation")
    ax.set(xlabel=r"Design $P_{fa}$", ylabel=r"Measured $P_{fa}$", xlim=(3e-7, 2e-2), ylim=(3e-7, 2e-2))
    ax.legend(loc="lower right", frameon=False, fontsize=6.5, handlelength=1.5)
    fig.tight_layout(); fig.savefig(FIG / "mc_pfa.png", dpi=300); plt.close(fig)

    s_fine = np.linspace(snr_map_db.min() - 1, snr_map_db.max() + 1, 200)
    fig, ax = plt.subplots(figsize=(3.6, 2.2))
    ax.plot(s_fine, pd_known_noise(10 ** (s_fine / 10), PFA_DETECT), color=INK2, lw=1, ls="--",
            label="Known noise level")
    ax.plot(s_fine, pd_ca_cfar(10 ** (s_fine / 10), PFA_DETECT, N_TRAIN), color=ORANGE, lw=1.4,
            label=f"CA-CFAR, N = {N_TRAIN}")
    ax.plot(snr_map_db, pd_meas, "o", color=BLUE, ms=4, label=f"Simulation ({N_TRIALS} trials)")
    ax.set(xlabel="SNR in the range–Doppler cell [dB]", ylabel=r"$P_d$", ylim=(-0.03, 1.03))
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "mc_pd.png", dpi=300); plt.close(fig)


if __name__ == "__main__":
    import sys
    plot() if "--plot-only" in sys.argv else main()
