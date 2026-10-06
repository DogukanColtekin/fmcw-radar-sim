"""A simulated FMCW radar, from the transmitted chirp to a list of targets.

We send chirps, build the echoes, mix them down, filter and sample them,
take a 2D FFT and let CFAR pick out the targets.

Run:  python fmcw_radar_sim.py      (figures go to ./figures)
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage, signal
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

C = 3e8


@dataclass
class Radar:
    fc: float = 77e9            # carrier [Hz]
    slope: float = 7.5e12       # chirp slope S [Hz/s]
    t_ramp: float = 24e-6       # ramp duration [s]
    t_idle: float = 6e-6        # pause between ramps [s]
    t_adc_start: float = 2e-6   # ADC waits this long after the ramp starts [s]
    t_adc: float = 20e-6        # how long the ADC samples [s]
    n_chirps: int = 128
    fs_sim: float = 200e6       # fine time grid that stands in for the analog signal
    fs_adc: float = 20e6        # ADC rate (IQ)
    f_cut: float = 8e6          # low-pass cutoff [Hz]
    snr_ref_db: float = -15.0   # SNR per sample of a 1 m^2 target at 100 m

    @property
    def wavelength(self): return C / self.fc
    @property
    def t_cri(self): return self.t_ramp + self.t_idle            # one ramp plus its pause
    @property
    def bw_eff(self): return self.slope * self.t_adc              # only what the ADC sees
    @property
    def range_res(self): return C / (2 * self.bw_eff)
    @property
    def range_max(self): return self.f_cut * C / (2 * self.slope)  # the filter decides this
    @property
    def v_max(self): return self.wavelength / (4 * self.t_cri)
    @property
    def v_res(self): return self.wavelength / (2 * self.n_chirps * self.t_cri)
    @property
    def decim(self): return int(round(self.fs_sim / self.fs_adc))


@dataclass
class Target:
    r0: float       # range at t = 0 [m]
    v: float        # [m/s], positive = moving away
    rcs: float      # [m^2]
    name: str = ""


TARGETS = [
    Target(18.0, -12.0, 1.0, "T1"),
    Target(47.0, 5.0, 3.0, "T2"),
    Target(48.5, -3.0, 2.0, "T3"),
    Target(85.0, 0.0, 20.0, "T4"),
    Target(122.0, 20.0, 10.0, "T5"),
    Target(150.0, -25.0, 5.0, "T6"),
    Target(470.0, 8.0, 2000.0, "Far"),   # too far to see; the filter should remove it
]


@dataclass
class Detection:
    r: float
    v: float
    snr_db: float


# ---------------------------------------------------------------- simulation

def antialias_filter(radar: Radar) -> np.ndarray:
    """Low-pass FIR: flat up to f_cut, about 80 dB down 4 MHz above it."""
    trans = 4e6
    numtaps, beta = signal.kaiserord(80, trans / (radar.fs_sim / 2))
    numtaps |= 1
    h = signal.firwin(numtaps, radar.f_cut + trans / 2, window=("kaiser", beta), fs=radar.fs_sim)
    return h / h.sum()


def simulate_beat(radar: Radar, targets: list[Target], h: np.ndarray, rng) -> np.ndarray:
    """Echoes plus noise, mixed with the TX chirp. One row per chirp, at fs_sim."""
    n_fast = int(round(radar.t_ramp * radar.fs_sim))
    t_fast = np.arange(n_fast) / radar.fs_sim
    t_slow = np.arange(radar.n_chirps)[:, None] * radar.t_cri
    f0 = -radar.slope * radar.t_ramp / 2                # ramp runs from -B/2 to +B/2

    def chirp_phase(u):
        return 2 * np.pi * (f0 * u + 0.5 * radar.slope * u**2)

    # scale the noise so its power is 1 once it has passed the filter
    sigma2_sim = 1.0 / np.sum(h**2)
    amp_ref = np.sqrt(10 ** (radar.snr_ref_db / 10))

    rx = np.zeros((radar.n_chirps, n_fast), complex)
    for tg in targets:
        tau = 2 * (tg.r0 + tg.v * (t_slow + t_fast)) / C     # delay changes as the target moves
        u = t_fast - tau
        valid = (u >= 0) & (u <= radar.t_ramp)
        amp = amp_ref * np.sqrt(tg.rcs) * (100.0 / tg.r0) ** 2
        phi = chirp_phase(u) - 2 * np.pi * radar.fc * tau + rng.uniform(0, 2 * np.pi)
        rx += valid * amp * np.exp(1j * phi)

    rx += np.sqrt(sigma2_sim / 2) * (rng.standard_normal(rx.shape) + 1j * rng.standard_normal(rx.shape))
    tx = np.exp(1j * chirp_phase(t_fast))
    return tx[None, :] * np.conj(rx)


def adc(radar: Radar, beat: np.ndarray, h: np.ndarray | None) -> np.ndarray:
    """Filter if h is given, then keep every decim-th sample of the ADC window."""
    x = signal.oaconvolve(beat, h[None, :], mode="same", axes=1) if h is not None else beat
    i0 = int(round(radar.t_adc_start * radar.fs_sim))
    n = int(round(radar.t_adc * radar.fs_adc))
    return x[:, i0 : i0 + n * radar.decim : radar.decim]


# ---------------------------------------------------------------- processing

def range_doppler(radar: Radar, x: np.ndarray, n_range_fft=512):
    """Hann-windowed FFT over range, then over Doppler. Returns the power map and both axes."""
    n_chirps, n_samp = x.shape
    x = x * signal.windows.hann(n_samp)[None, :]
    rng_fft = np.fft.fft(x, n_range_fft, axis=1)[:, : n_range_fft // 2]   # targets only show up at positive beats
    rng_fft *= signal.windows.hann(n_chirps)[:, None]
    rd = np.fft.fftshift(np.fft.fft(rng_fft, axis=0), axes=0).T          # (range, doppler)
    power = np.abs(rd) ** 2

    f_beat = np.arange(n_range_fft // 2) * radar.fs_adc / n_range_fft
    f_dopp = np.fft.fftshift(np.fft.fftfreq(n_chirps, radar.t_cri))
    return power, f_beat, f_dopp


def ca_cfar_2d(power, guard=(2, 2), train=(8, 4), pfa=1e-6):
    """Compare each cell with the mean of its neighbours. Doppler wraps, range is mirrored at the edges."""
    gr, gd = guard
    tr, td = train
    kr, kd = gr + tr, gd + td
    kernel = np.ones((2 * kr + 1, 2 * kd + 1))
    kernel[tr : tr + 2 * gr + 1, td : td + 2 * gd + 1] = 0
    n_train = kernel.sum()

    padded = np.pad(power, ((kr, kr), (0, 0)), mode="reflect")
    padded = np.pad(padded, ((0, 0), (kd, kd)), mode="wrap")
    noise = signal.fftconvolve(padded, kernel, mode="valid") / n_train
    alpha = n_train * (pfa ** (-1 / n_train) - 1)   # valid when noise power is exponential
    threshold = alpha * noise
    return power > threshold, threshold, noise


def _parabolic(y_m, y_0, y_p):
    den = y_m - 2 * y_0 + y_p
    return 0.0 if den == 0 else 0.5 * (y_m - y_p) / den


def extract_detections(radar, power, mask, noise, f_beat, f_dopp, min_range_bin=3):
    """Take the peak of each blob of CFAR hits and interpolate it between bins."""
    mask = mask.copy()
    mask[:min_range_bin, :] = False
    labels, n = ndimage.label(mask)
    df_b = f_beat[1] - f_beat[0]
    df_d = f_dopp[1] - f_dopp[0]
    n_r, n_d = power.shape
    log_p = 10 * np.log10(power + 1e-30)

    dets = []
    for k in range(1, n + 1):
        cells = np.argwhere(labels == k)
        ir, idp = cells[np.argmax(power[labels == k])]
        dr = _parabolic(*log_p[max(ir - 1, 0) : ir + 2, idp]) if 0 < ir < n_r - 1 else 0.0
        dd = _parabolic(log_p[ir, (idp - 1) % n_d], log_p[ir, idp], log_p[ir, (idp + 1) % n_d])
        fd = f_dopp[idp] + dd * df_d
        fb = f_beat[ir] + dr * df_b
        r = C / (2 * radar.slope) * (fb - fd)        # Doppler shifts the beat too, undo it
        v = fd * radar.wavelength / 2
        dets.append(Detection(r, v, 10 * np.log10(power[ir, idp] / noise[ir, idp])))
    return sorted(dets, key=lambda d: d.r)


def match(targets, dets, gate=(3.0, 2.0)):
    rows = []
    for tg in targets:
        best = min(dets, key=lambda d: abs(d.r - tg.r0) / gate[0] + abs(d.v - tg.v) / gate[1], default=None)
        ok = best is not None and abs(best.r - tg.r0) < gate[0] and abs(best.v - tg.v) < gate[1]
        rows.append((tg, best if ok else None))
    used = {id(d) for _, d in rows if d is not None}
    extras = [d for d in dets if id(d) not in used]
    return rows, extras


# ---------------------------------------------------------------- plotting

INK, INK2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
SEQ = LinearSegmentedColormap.from_list("seq_blue", [SURFACE, "#a9c8ef", BLUE, "#123f78", "#071a33"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold", "lines.linewidth": 1.5,
})


def plot_beat_spectrum(radar, beat, h, path):
    x = beat[0] * signal.windows.hann(beat.shape[1])
    nfft = 1 << 15
    f = np.fft.fftshift(np.fft.fftfreq(nfft, 1 / radar.fs_sim))
    X = np.fft.fftshift(np.fft.fft(x, nfft))
    X_db = 20 * np.log10(np.abs(X) / np.abs(X).max() + 1e-12)
    H_db = 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(h, nfft))) + 1e-12)

    fig, ax = plt.subplots(figsize=(8, 4.0))
    ax.plot(f / 1e6, X_db, color=BLUE, lw=0.8, label="Beat spectrum (one chirp, before filter)")
    ax.plot(f / 1e6, H_db, color=ORANGE, lw=2, label="Anti-alias low-pass |H(f)|")
    ax.axvspan(radar.fs_adc / 2 / 1e6, 50, color=GRID, alpha=0.5, lw=0)
    ax.text(10.6, 2, "aliases into\nADC band", color=INK2, fontsize=8, va="top")
    f_far = 2 * 470 * radar.slope / C / 1e6
    ax.annotate("far reflector (470 m)", xy=(f_far, -22), xytext=(f_far + 2, -8), color=INK2, fontsize=8,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.set(xlim=(0, 40), ylim=(-110, 5), xlabel="Beat frequency [MHz]", ylabel="dB",
           title="Dechirped signal: in-band targets vs. far reflector at 23.5 MHz")
    top = ax.secondary_xaxis("top", functions=(lambda fm: fm * 1e6 * C / (2 * radar.slope),
                                                lambda r: r * 2 * radar.slope / C / 1e6))
    top.set_xlabel("Equivalent range [m]", color=INK2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def plot_range_profile(radar, x_adc, targets, path):
    n_fft = 512
    xw = x_adc[0] * signal.windows.hann(x_adc.shape[1])
    prof = np.abs(np.fft.fft(xw, n_fft)[: n_fft // 2]) ** 2
    rng_axis = np.arange(n_fft // 2) * radar.fs_adc / n_fft * C / (2 * radar.slope)
    noise_floor = np.median(prof)

    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(rng_axis, 10 * np.log10(prof / noise_floor), color=BLUE)
    labelled = []
    for tg in sorted(targets, key=lambda t: t.r0):
        if tg.r0 < radar.range_max:
            ax.axvline(tg.r0, color=INK2, lw=0.8, ls=":")
            if labelled and tg.r0 - labelled[-1][0] < 5:
                labelled[-1][1].append(tg.name)
            else:
                labelled.append((tg.r0, [tg.name]))
    for r, names in labelled:
        ax.text(r, 1.0, " " + ", ".join(names), transform=ax.get_xaxis_transform(), color=INK2, fontsize=8, va="top")
    ax.set(xlim=(0, 200), xlabel="Range [m]", ylabel="dB over noise floor",
           title="Range profile of a single chirp (fast-time FFT)")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def plot_rd_maps(radar, maps, targets, path):
    fig, axes = plt.subplots(1, len(maps), figsize=(6.2 * len(maps), 5.2), squeeze=False)
    for ax, (title, power, noise_ref, f_beat, f_dopp, dets) in zip(axes[0], maps):
        rng_axis = f_beat * C / (2 * radar.slope)
        vel_axis = f_dopp * radar.wavelength / 2
        img = 10 * np.log10(power / noise_ref)
        im = ax.imshow(img, origin="lower", aspect="auto", cmap=SEQ, vmin=-5, vmax=50,
                       extent=[vel_axis[0], vel_axis[-1], rng_axis[0], rng_axis[-1]])
        ax.grid(False)
        for tg in targets:
            if tg.r0 < radar.range_max:
                ax.plot(tg.v, tg.r0, marker="+", ms=14, mew=1.5, color=INK, ls="")
        ax.plot([d.v for d in dets], [d.r for d in dets], "o", mfc="none", mec=ORANGE, ms=13, mew=2, ls="")
        ax.set(title=title, xlabel="Radial velocity [m/s]  (+ receding)", ylabel="Range [m]")
        fig.colorbar(im, ax=ax, label="dB over median noise (filtered case)", shrink=0.9)
    h1 = plt.Line2D([], [], marker="+", ms=12, mew=1.5, color=INK, ls="", label="Ground truth")
    h2 = plt.Line2D([], [], marker="o", mfc="none", mec=ORANGE, ms=10, mew=2, ls="", label="CFAR detection")
    fig.legend(handles=[h1, h2], loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(path, dpi=160); plt.close(fig)


def plot_cfar_cut(radar, power, threshold, noise_ref, f_beat, f_dopp, v_cut, path):
    idp = np.argmin(np.abs(f_dopp * radar.wavelength / 2 - v_cut))
    rng_axis = f_beat * C / (2 * radar.slope)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(rng_axis, 10 * np.log10(power[:, idp] / noise_ref), color=BLUE, label="Cell power")
    ax.plot(rng_axis, 10 * np.log10(threshold[:, idp] / noise_ref), color=ORANGE, lw=2, label="CA-CFAR threshold")
    ax.set(xlim=(0, 200), xlabel="Range [m]", ylabel="dB over noise floor",
           title=f"CFAR along range at v = {f_dopp[idp] * radar.wavelength / 2:+.1f} m/s")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


# ---------------------------------------------------------------- main

def main(seed=7, out_dir=Path(__file__).resolve().parent / "figures"):
    out_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    radar = Radar()
    targets = TARGETS

    print(f"Range res {radar.range_res:.2f} m | R_max {radar.range_max:.0f} m | "
          f"v res {radar.v_res:.2f} m/s | v_max +/-{radar.v_max:.1f} m/s")

    h = antialias_filter(radar)
    beat = simulate_beat(radar, targets, h, rng)

    results = {}
    for label, filt in [("with anti-alias filter", h), ("without filter", None)]:
        x = adc(radar, beat, filt)
        power, f_beat, f_dopp = range_doppler(radar, x)
        mask, thr, noise = ca_cfar_2d(power)
        dets = extract_detections(radar, power, mask, noise, f_beat, f_dopp)
        results[label] = dict(x=x, power=power, thr=thr, noise=noise, f_beat=f_beat,
                              f_dopp=f_dopp, dets=dets, floor=np.median(power))

    for label, res in results.items():
        rows, extras = match(TARGETS, res["dets"])
        print(f"\n--- {label}: {len(res['dets'])} detections ---")
        print(f"{'':5}{'R true':>8}{'R est':>8}{'dR':>7}{'v true':>9}{'v est':>8}{'dv':>7}{'SNR':>7}")
        for tg, d in rows:
            if d is None:
                print(f"{tg.name:5}{tg.r0:8.1f}{'miss':>8}{'':7}{tg.v:9.1f}")
            else:
                print(f"{tg.name:5}{tg.r0:8.1f}{d.r:8.2f}{d.r - tg.r0:7.2f}{tg.v:9.1f}{d.v:8.2f}"
                      f"{d.v - tg.v:7.2f}{d.snr_db:6.1f}dB")
        for d in extras:
            print(f"{'extra':5}{'':8}{d.r:8.2f}{'':7}{'':9}{d.v:8.2f}{'':7}{d.snr_db:6.1f}dB")

    a, b = results["with anti-alias filter"], results["without filter"]
    floor = a["floor"]
    print(f"\nNoise floor without filter is {10*np.log10(b['floor']/floor):.1f} dB higher")

    plot_beat_spectrum(radar, beat, h, out_dir / "1_beat_spectrum.png")
    plot_range_profile(radar, a["x"], targets, out_dir / "2_range_profile.png")
    plot_rd_maps(radar, [
        ("Range-Doppler map, with anti-alias filter", a["power"], floor, a["f_beat"], a["f_dopp"], a["dets"]),
        ("Same data, no filter (decimation only)", b["power"], floor, b["f_beat"], b["f_dopp"], b["dets"]),
    ], targets, out_dir / "3_range_doppler.png")
    plot_cfar_cut(radar, a["power"], a["thr"], floor, a["f_beat"], a["f_dopp"], 0.0, out_dir / "4_cfar_cut.png")


if __name__ == "__main__":
    main()
