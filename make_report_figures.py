"""Figures of the work

Run:  python make_report_figures.py      (figures go to ./figures/report)
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import signal

import fmcw_radar_sim as m
from fmcw_radar_sim import C, BLUE, ORANGE, INK, INK2, GRID, SEQ

OUT = Path(__file__).resolve().parent / "figures" / "report"
OUT.mkdir(parents=True, exist_ok=True)
WHITE = "#ffffff"
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "lines.linewidth": 1.2,
    "figure.facecolor": WHITE, "axes.facecolor": WHITE, "savefig.facecolor": WHITE,
    "axes.titleweight": "normal",
})
DPI = 300


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=DPI)
    plt.close(fig)


radar = m.Radar()
rng = np.random.default_rng(7)
h = m.antialias_filter(radar)
beat = m.simulate_beat(radar, m.TARGETS, h, rng)

res = {}
for key, filt in [("filt", h), ("raw", None)]:
    x = m.adc(radar, beat, filt)
    power, f_beat, f_dopp = m.range_doppler(radar, x)
    mask, thr, noise = m.ca_cfar_2d(power)
    dets = m.extract_detections(radar, power, mask, noise, f_beat, f_dopp)
    res[key] = dict(x=x, power=power, thr=thr, dets=dets)
floor = np.median(res["filt"]["power"])
rax = f_beat * C / (2 * radar.slope)
vax = f_dopp * radar.wavelength / 2
in_range = [t for t in m.TARGETS if t.r0 < radar.range_max]

# frame timing
S, Tr, Tc = radar.slope, radar.t_ramp, radar.t_cri
B = S * Tr / 1e6
fig, ax = plt.subplots(figsize=(4.6, 2.3))
for k in range(3):
    t0 = k * Tc * 1e6
    ax.axvspan(t0 + 2, t0 + 22, color=GRID, lw=0, label="ADC window" if k == 0 else None)
    ax.plot([t0, t0 + Tr * 1e6], [-B / 2, B / 2], color=BLUE, label="TX chirp" if k == 0 else None)
    ax.plot([t0 + Tr * 1e6, t0 + Tr * 1e6, t0 + Tc * 1e6], [B / 2, -B / 2, -B / 2], color=BLUE, lw=0.8, ls=":")
ax.annotate("", xy=(0, 108), xytext=(30, 108), arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.8))
ax.text(15, 114, r"$T_c = 30\,\mu$s", ha="center", va="bottom", color=INK2)
ax.text(93, 0, "×128", color=INK2, va="center")
ax.set(xlim=(-2, 102), ylim=(-110, 135), xlabel=r"Time [$\mu$s]", ylabel="Frequency offset [MHz]")
ax.legend(loc="lower right", frameon=False)
save(fig, "frame_timing.png")

# beat frequency geometry
tau = 2 * 150 / C * 1e6
tt = np.linspace(0, 6, 200)
fig, ax = plt.subplots(figsize=(3.6, 2.4))
ax.plot(tt, -B / 2 + S * tt * 1e-12, color=BLUE, label="TX")
u = tt[tt >= tau]
ax.plot(u, -B / 2 + S * (u - tau) * 1e-12, color=ORANGE, label="RX, R = 150 m")
ti = 4.0
f_tx, f_rx = -B / 2 + S * ti * 1e-12, -B / 2 + S * (ti - tau) * 1e-12
ax.annotate("", xy=(ti, f_rx), xytext=(ti, f_tx), arrowprops=dict(arrowstyle="<->", color=INK, lw=0.8))
ax.text(ti + 0.12, (f_tx + f_rx) / 2, r"$f_b$", va="center")
y0 = -B / 2 + S * 2.2e-12
ax.annotate("", xy=(2.2, y0), xytext=(2.2 + tau, y0), arrowprops=dict(arrowstyle="<->", color=INK, lw=0.8))
ax.text(2.7, y0 + 2, r"$\tau$", ha="center", va="bottom")
ax.set(xlim=(0, 6), xlabel=r"Time [$\mu$s]", ylabel="Frequency offset [MHz]")
ax.legend(loc="upper left", frameon=False)
save(fig, "beat_geometry.png")

# beat spectrum and filter response
xw = beat[0] * signal.windows.hann(beat.shape[1])
nfft = 1 << 15
f = np.fft.fftshift(np.fft.fftfreq(nfft, 1 / radar.fs_sim)) / 1e6
X = np.abs(np.fft.fftshift(np.fft.fft(xw, nfft)))
H = np.abs(np.fft.fftshift(np.fft.fft(h, nfft)))
fig, ax = plt.subplots(figsize=(4.6, 2.3))
ax.axvspan(radar.fs_adc / 2e6, 40, color=GRID, lw=0)
ax.plot(f, 20 * np.log10(X / X.max() + 1e-12), color=BLUE, lw=0.6, label="Dechirped signal")
ax.plot(f, 20 * np.log10(H + 1e-12), color=ORANGE, lw=1.4, label="Filter response")
ax.annotate("470 m", xy=(23.5, -23), xytext=(26, -10), color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
ax.text(10.6, -6, "aliased band", color=INK2, fontsize=7, va="top")
ax.set(xlim=(0, 40), ylim=(-110, 5), xlabel="Beat frequency [MHz]", ylabel="Magnitude [dB]")
ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.27), frameon=False)
save(fig, "beat_spectrum.png")

# range-Doppler maps with and without the filter
fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.5), sharey=True)
for ax, key, title in [(axes[0], "filt", "With filter"), (axes[1], "raw", "Without filter")]:
    img = 10 * np.log10(res[key]["power"] / floor)
    im = ax.imshow(img, origin="lower", aspect="auto", cmap=SEQ, vmin=-5, vmax=50,
                   extent=[vax[0], vax[-1], rax[0], rax[-1]])
    ax.grid(False)
    d = res[key]["dets"]
    ax.plot([p.v for p in d], [p.r for p in d], "o", mfc="none", mec=ORANGE, ms=6, mew=1.2)
    ax.set(title=title, xlabel="v [m/s]")
axes[0].set_ylabel("R [m]")
fig.colorbar(im, ax=axes, label="dB", shrink=0.9, pad=0.03)
fig.savefig(OUT / "rd_compare.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)

# data matrix
xr = res["filt"]["x"].real
lim = np.percentile(np.abs(xr), 99)
fig, ax = plt.subplots(figsize=(3.4, 2.5))
ax.imshow(xr, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim,
          extent=[0, xr.shape[1], 0, xr.shape[0]])
ax.grid(False)
ax.set(xlabel="Fast-time sample n", ylabel="Chirp index m")
save(fig, "data_matrix.png")

# range profile of one chirp
n_fft = 512
xw = res["filt"]["x"][0] * signal.windows.hann(res["filt"]["x"].shape[1])
prof = np.abs(np.fft.fft(xw, n_fft)[: n_fft // 2]) ** 2
fig, ax = plt.subplots(figsize=(4.6, 1.9))
ax.plot(rax, 10 * np.log10(prof / np.median(prof)), color=BLUE)
for t in in_range:
    ax.axvline(t.r0, color=INK2, lw=0.6, ls=":")
for r, lab in [(18, "T1"), (47.75, "T2, T3"), (85, "T4"), (122, "T5"), (150, "T6")]:
    ax.text(r + 1.5, 42, lab, color=INK2, fontsize=7, va="top")
ax.set(xlim=(0, 200), ylim=(-25, 45), xlabel="Range [m]", ylabel="dB")
save(fig, "range_profile.png")

# labelled range-Doppler map
fig, ax = plt.subplots(figsize=(4.2, 3.0))
im = ax.imshow(10 * np.log10(res["filt"]["power"] / floor), origin="lower", aspect="auto", cmap=SEQ,
               vmin=-5, vmax=50, extent=[vax[0], vax[-1], rax[0], rax[-1]])
ax.grid(False)
for d in res["filt"]["dets"]:
    ax.plot(d.v, d.r, "o", mfc="none", mec=ORANGE, ms=8, mew=1.3)
offs = {"T1": (2.5, 0), "T2": (2.5, -6), "T3": (-2.5, 6), "T4": (2.5, 0), "T5": (2.5, 0), "T6": (2.5, 0)}
for t in in_range:
    dx, dy = offs[t.name]
    ax.text(t.v + dx, t.r0 + dy, t.name, fontsize=7.5, va="center", ha="left" if dx > 0 else "right")
ax.set(xlabel="Radial velocity [m/s]", ylabel="Range [m]")
fig.colorbar(im, ax=ax, label="dB above noise floor", pad=0.02)
save(fig, "rd_map.png")

# CFAR window
gr, gd, tr, td = 2, 2, 8, 4
nr, nd = 2 * (gr + tr) + 1, 2 * (gd + td) + 1
fig, ax = plt.subplots(figsize=(2.0, 2.9))
for i in range(nr):
    for j in range(nd):
        di, dj = abs(i - nr // 2), abs(j - nd // 2)
        c = ORANGE if di == dj == 0 else "#cfcfcf" if (di <= gr and dj <= gd) else "#a9c8ef"
        ax.add_patch(Rectangle((j, i), 1, 1, facecolor=c, edgecolor=WHITE, lw=0.8))
ax.set(xlim=(0, nd), ylim=(0, nr), aspect="equal", xticks=[], yticks=[])
ax.grid(False)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_xlabel("Doppler")
ax.set_ylabel("Range")
save(fig, "cfar_window.png")

# CFAR threshold along range at v = 0
idp = np.argmin(np.abs(vax))
fig, ax = plt.subplots(figsize=(4.6, 2.1))
ax.plot(rax, 10 * np.log10(res["filt"]["power"][:, idp] / floor), color=BLUE, lw=0.9, label="Cell power")
ax.plot(rax, 10 * np.log10(res["filt"]["thr"][:, idp] / floor), color=ORANGE, lw=1.4, label="Threshold")
ax.set(xlim=(0, 200), ylim=(-40, 55), xlabel="Range [m]", ylabel="dB")
ax.legend(loc="upper right", frameon=False, ncol=2)
save(fig, "cfar_cut.png")
