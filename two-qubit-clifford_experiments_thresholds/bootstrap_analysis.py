# Thanks Timo Hillmann for this code!
from dataclasses import dataclass
import numpy as np
import scipy.stats as stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import scipy
import io
import os
import csv
import json
from typing import Any, Optional


def distribution_setup(
    fail: np.ndarray,
    shots: np.ndarray,
    CI_val: float = 0.025,
):
    succ = shots - fail

    Ds = []
    μs = []
    σs = []
    CI_ls = []
    CI_hs = []
    for i in range(fail.shape[0]):
        _Ds = []
        _μs = []
        _σs = []
        _CI_ls = []
        _CI_hs = []
        for j in range(len(fail[i])):
            _Ds.append(stats.beta(fail[i][j] + 1, succ[i][j] + 1))
            _μs.append(_Ds[-1].mean())
            _σs.append(_Ds[-1].std())
            _CI_ls.append(_Ds[-1].ppf(CI_val))
            _CI_hs.append(_Ds[-1].ppf(1 - CI_val))

        Ds.append(_Ds)
        μs.append(_μs)
        σs.append(_σs)
        CI_ls.append(_CI_ls)
        CI_hs.append(_CI_hs)

    return μs, σs, CI_ls, CI_hs, Ds


def _distribution_setup(
    fail: np.ndarray,
    shots: np.ndarray,
    CI_val: float = 0.025,
):
    succ = shots - fail

    D = stats.beta(fail + 1, succ + 1)
    Ds = np.zeros_like(fail, dtype=object)
    for i, j in np.ndindex(Ds.shape):
        Ds[i, j] = stats.beta(fail[i, j] + 1, succ[i, j] + 1)

    μ = D.mean()
    σ = D.std()
    CI_l = D.ppf(CI_val)
    CI_h = D.ppf(1 - CI_val)

    return μ, σ, CI_l, CI_h, Ds


def rescale_input(p, p_th, mu):
    return (p[0] - p_th) * p[1] ** (2 / mu)


def fit_func(x, *args):
    # return scipy.special.betainc(args[0], args[1], x)
    # return (1 + np.tanh((x * args[0]))) * 0.5
    return sum([a * x**i for i, a in enumerate(args)])


def rescaled_fit_func(p, p_th, mu, *args):
    return fit_func(rescale_input(p, p_th, mu), *args)


def least_square_fit(ps, ds, ys, dys=None, order=2, p0=None):

    _p0 = np.ones(order + 3)
    _p0[0] = np.mean(ps)
    if p0 is not None:
        for idx, val in enumerate(p0):
            _p0[idx] = val

    fit = curve_fit(
        rescaled_fit_func,
        (ps, ds),
        ys,
        p0=_p0,
        sigma=dys if dys is not None else None,
        maxfev=100_000,
        absolute_sigma=True,
    )

    return fit


def multiround_least_square_fit(ps, ds, ys, rounds=1, dys=None, p0=None):

    # define as closure to pass the number of rounds
    def multi_round_fit_func(x, *args):
        return args[1] * (1 - (1 - (1 + np.tanh(x * args[0])) / 2) ** args[2])

    def rescaled_multi_round_fit_func(p, p_th, mu, *args):
        return multi_round_fit_func(rescale_input(p, p_th, mu), *args)

    _p0 = np.ones(2 + 3)
    _p0[0] = np.mean(ps)
    if p0 is not None:
        for idx, val in enumerate(p0):
            _p0[idx] = val

    fit = curve_fit(
        rescaled_multi_round_fit_func,
        (ps, ds),
        ys,
        p0=_p0,
        sigma=dys if dys is not None else None,
        maxfev=1_000_000,
        absolute_sigma=True,
    )

    return fit


def multiround_boostrap_resample(
    ps: np.ndarray,
    ds: np.ndarray,
    Ds: np.ndarray,
    σ: np.ndarray,
    fit_params: np.ndarray,
    rounds: int = 1,
    n_samples: int = 1000,
    order: int = 2,
    CI_val: float = 0.025,
):
    resamples = np.zeros(n_samples, dtype=np.float64)
    for idx in range(n_samples):
        ry = []
        for i in range(len(Ds)):
            for j in range(len(Ds[i])):
                ry.append(Ds[i][j].rvs())

        fit = multiround_least_square_fit(
            ps, ds, ry, rounds=rounds, dys=σ, p0=fit_params
        )

        resamples[idx] = fit[0][0]

    resamples = np.sort(resamples)
    mean = np.mean(resamples)
    Δ = np.std(resamples, ddof=1)
    CI_l = resamples[int(n_samples * CI_val)]
    CI_h = resamples[int(n_samples * (1 - CI_val))]
    return mean, Δ, CI_l, CI_h, resamples


def boostrap_resample(
    ps: np.ndarray,
    ds: np.ndarray,
    Ds: np.ndarray,
    σ: np.ndarray,
    fit_params: np.ndarray,
    n_samples: int = 1000,
    order: int = 2,
    CI_val: float = 0.025,
):
    resamples = np.zeros(n_samples, dtype=np.float64)
    for idx in range(n_samples):
        ry = []
        for i in range(len(Ds)):
            for j in range(len(Ds[i])):
                ry.append(Ds[i][j].rvs())

        fit = least_square_fit(ps, ds, ry, dys=σ, order=order, p0=fit_params)

        resamples[idx] = fit[0][0]

    resamples = np.sort(resamples)
    mean = np.mean(resamples)
    Δ = np.std(resamples, ddof=1)
    CI_l = resamples[int(n_samples * CI_val)]
    CI_h = resamples[int(n_samples * (1 - CI_val))]
    return mean, Δ, CI_l, CI_h, resamples


def bootstrap_plot(
    ps: np.ndarray,
    ds: np.ndarray,
    fail: np.ndarray,
    shots: np.ndarray,
    order: int = 2,
    weighted: bool = True,
    CI_val: float = 0.025,
    n_samples: int = 1000,
    fig: plt.Figure = None,
    ax: plt.Axes = None,
    **kwargs,
):

    μ, σ, CI_l, CI_h, Ds = distribution_setup(fail, shots, CI_val=CI_val)

    _μ = np.hstack(μ)
    _σ = np.hstack(σ) if weighted else None
    _ds = np.hstack(ds)
    _ps = np.hstack(ps)

    fit_params, chi = least_square_fit(_ps, _ds, _μ, dys=_σ, order=order)

    mean_LS = fit_params[0]
    Δ_LS = np.sqrt(chi[0, 0])

    mean, Δ, CI_l, CI_h, resamples = boostrap_resample(
        _ps, _ds, Ds, _σ, fit_params, n_samples=n_samples, order=order, CI_val=CI_val
    )

    print("Standard Error Analysis")
    print(f"τ_LF = ({100 *mean_LS:.4f} ± {196 * Δ_LS:.4f})%")

    print("Bootstrap Analysis")
    print(f"τ_BS  = ({100 * mean:.4f} ± {196 * Δ:.4f})%")
    print(f"CI_BS = ({100 * CI_l:.4f}, {100 * CI_h:.4f})%")
    print(f"      = ({100 * (CI_l - mean):.4f}, {100 * (CI_h - mean):.4f})%")

    if fig is None:
        fig, axis = plt.subplots()

    if isinstance(ax, np.ndarray):
        axis = ax[0]

    axis.hist(resamples, bins=50, density=True, alpha=0.33, color="grey")
    axis.axvline(mean_LS, color="C0", label="Least Squares Fit", lw=1)
    axis.axvline(mean, color="C1", label="Bootstrap Estimate", lw=1)

    # dra lines for the Least Square Fit confidence interval
    axis.axvline(mean_LS - Δ_LS, color="C0", linestyle="--", lw=0.5)
    axis.axvline(mean_LS + Δ_LS, color="C0", linestyle="--", lw=0.5)

    # shade the confidence interval of the bootstrap
    axis.axvspan(CI_l, CI_h, color="black", alpha=0.2, zorder=-1)
    axis.set_xlim(mean - 2 * (mean - CI_l), mean + 2 * (CI_h - mean))

    axis.set_xlabel(r"$p_{\rm th}$")
    axis.set_ylabel("Density")
    axis.legend()

    _, _ = collapse_plot(ps, μ, ds, fit_params, fig, ax[1])

    return (
        fig,
        ax,
        ThresholdData(
            1,
            1,
            mean,
            CI_l,
            CI_h,
            {
                "bootstrap_samples": n_samples,
                "weighted": weighted,
                "fit_params": fit_params.tolist(),
            },
        ),
    )


@dataclass
class ThresholdData:
    rounds: int
    window: int
    threshold: float
    conf_low: float
    conf_high: float
    json_metadata: dict


def multiround_bootstrap_plot(
    ps: np.ndarray,
    ds: np.ndarray,
    fail: np.ndarray,
    shots: np.ndarray,
    rounds: int = 1,
    order: int = 2,
    weighted: bool = True,
    CI_val: float = 0.025,
    n_samples: int = 1000,
    fig: plt.Figure = None,
    ax: plt.Axes = None,
    filepath: str = None,
    window: int = 1,
    p0: Optional[list] = None,
    **kwargs,
):

    μ, σ, CI_l, CI_h, Ds = distribution_setup(fail, shots, CI_val=CI_val)

    _μ = np.hstack(μ)
    _σ = np.hstack(σ) if weighted else None
    _ds = np.hstack(ds)
    _ps = np.hstack(ps)

    fit_params, chi = multiround_least_square_fit(
        _ps, _ds, _μ, rounds=rounds, dys=_σ, p0=p0
    )

    mean_LS = fit_params[0]
    Δ_LS = np.sqrt(chi[0, 0])

    mean, Δ, CI_l, CI_h, resamples = multiround_boostrap_resample(
        _ps,
        _ds,
        Ds,
        _σ,
        fit_params,
        rounds=rounds,
        n_samples=n_samples,
        order=order,
        CI_val=CI_val,
    )

    print("Standard Error Analysis")
    print(f"τ_LF = ({100 *mean_LS:.4f} ± {196 * Δ_LS:.4f})%")

    print("Bootstrap Analysis")
    print(f"τ_BS  = ({100 * mean:.4f} ± {196 * Δ:.4f})%")
    print(f"CI_BS = ({100 * CI_l:.4f}, {100 * CI_h:.4f})%")
    print(f"      = ({100 * (CI_l - mean):.4f}, {100 * (CI_h - mean):.4f})%")

    if fig is None:
        fig, ax = plt.subplots()

    if isinstance(ax, np.ndarray):
        axis = ax[0]
    else:
        axis = ax

    axis.hist(resamples, bins=50, density=True, alpha=0.33, color="grey")
    axis.axvline(mean_LS, color="C0", label="Least Sq.", lw=1)
    axis.axvline(mean, color="C1", label="Bootstrap", lw=1)

    # dra lines for the Least Square Fit confidence interval
    axis.axvline(mean_LS - Δ_LS, color="C0", linestyle="--", lw=0.5)
    axis.axvline(mean_LS + Δ_LS, color="C0", linestyle="--", lw=0.5)

    # shade the confidence interval of the bootstrap
    axis.axvspan(CI_l, CI_h, color="C1", alpha=0.2, zorder=-1)
    axis.set_xlim(mean - 2 * (mean - CI_l), mean + 2 * (CI_h - mean))

    axis.set_xlabel(r"$p_{\rm th}$")
    axis.set_ylabel("Density")
    #axis.legend(
    #    loc="upper center",
    #    ncol=2,
    #    bbox_to_anchor=(0.5, 1.125),
    #    columnspacing=0.85,
    #    handletextpad=0.55,
    #    fontsize=7,
    #)
    axis.legend(loc="upper right", fontsize=7)

    if (np.max(resamples) - np.min(resamples)) < 1e-5:
        axis.set_xlim(mean - 1e-5, mean + 1e-5)

    _, _ = multiround_collapse_plot(
        ps, μ, ds, fit_params, rounds=rounds, fig=fig, ax=ax[1]
    )

    save_fit_results(
        filepath,
        rounds,
        window,
        mean,
        CI_l,
        CI_h,
        {
            "bootstrap_samples": n_samples,
            "weighted": weighted,
            "fit_params": fit_params.tolist(),
        },
    )

    return (
        fig,
        ax,
        ThresholdData(
            rounds,
            window,
            mean,
            CI_l,
            CI_h,
            {
                "bootstrap_samples": n_samples,
                "weighted": weighted,
                "fit_params": fit_params.tolist(),
            },
        ),
    )


def multiround_collapse_plot(
    ps: np.ndarray,
    ys: np.ndarray,
    ds: np.ndarray,
    fit_params: np.ndarray,
    rounds: int = 1,
    fig: plt.Figure = None,
    ax: plt.Axes = None,
):

    # define as closure to pass the number of rounds
    def multi_round_fit_func(x, *args):
        return args[1] * (1 - (1 - (1 + np.tanh(x * args[0])) / 2) ** args[2])

    if fig is None:
        fig, ax = plt.subplots()

    xmin = 0.0
    xmax = 0.0
    for idx in range(ps.shape[0]):
        _x = rescale_input((ps[idx], ds[idx]), *fit_params[:2])
        ax.plot(
            _x,
            ys[idx],
            label=f"$d={ds[idx][0]}$",
            ls="none",
            marker="o",
            ms=3,
        )
        xmin = min(xmin, _x.min())
        xmax = max(xmax, _x.max())

    x = np.linspace(xmin, xmax, 512)
    y = multi_round_fit_func(x, *fit_params[2:])
    ax.plot(x, y, label="Fit", color="black", lw=0.5, zorder=0)

    ax.set_xlabel(r"$ (p - p_{\rm th}) d^{1/\mu}$")
    ax.set_ylabel(r"$p_{\rm err}$")
    ax.set_title("Collapse Plot", fontsize=7, pad=2)
    ax.legend()
    return fig, ax


def collapse_plot(
    ps: np.ndarray,
    ys: np.ndarray,
    ds: np.ndarray,
    fit_params: np.ndarray,
    fig: plt.Figure = None,
    ax: plt.Axes = None,
):

    if fig is None:
        fig, ax = plt.subplots()

    xmin = 0.0
    xmax = 0.0
    for idx in range(ps.shape[0]):
        _x = rescale_input((ps[idx], ds[idx]), *fit_params[:2])
        ax.plot(
            _x,
            ys[idx],
            label=f"$d={ds[idx][0]}$",
            ls="none",
            marker="o",
            ms=3,
        )
        xmin = min(xmin, _x.min())
        xmax = max(xmax, _x.max())

    x = np.linspace(xmin, xmax, 512)
    y = fit_func(x, *fit_params[2:])
    ax.plot(x, y, label="Fit", color="black", lw=0.5, zorder=0)

    ax.set_xlabel(r"$ (p - p_{\rm th}) d^{1/\mu}$")
    ax.set_ylabel(r"$p_{\rm err}$")
    ax.legend()
    return fig, ax


def save_fit_results(
    path: str,
    rounds: int,
    window: int,
    mean: float,
    CI_l: float,
    CI_h: float,
    json_metadata: dict,
):
    if path is None:
        return

    # check if the file exists and if not create it
    os.makedirs(path, exist_ok=True)
    fn = "fit_results_weighted.csv" if json_metadata["weighted"] else "fit_results.csv"
    path = os.path.join(path, fn)
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(CSV_HEADER)

    with open(path, "a") as f:
        f.write(
            "\n"
            + csv_line(
                rounds=rounds,
                window=window,
                threshold=mean,
                conf_low=CI_l,
                conf_high=CI_h,
                json_metadata=json_metadata,
            )
        )


def escape_csv(text: Any, width: Optional[int]) -> str:
    output = io.StringIO()
    csv.writer(output).writerow([text])
    text = output.getvalue().strip()
    if width is not None:
        text = text.rjust(width)
    return text


def csv_line(
    *,
    rounds: Any,
    window: Any,
    threshold: Any,
    conf_low: Any,
    conf_high: Any,
    json_metadata: Any,
    is_header: bool = False,
) -> str:

    if not is_header:
        json_metadata = json.dumps(json_metadata, separators=(",", ":"), sort_keys=True)

        # round to 6 decimal places
        threshold = f"{threshold:.6f}"
        conf_low = f"{conf_low:.6f}"
        conf_high = f"{conf_high:.6f}"

    rounds = escape_csv(rounds, 3)
    window = escape_csv(window, 2)
    threshold = escape_csv(threshold, 9)
    conf_low = escape_csv(conf_low, 9)
    conf_high = escape_csv(conf_high, 9)
    json_metadata = escape_csv(json_metadata, None)

    return (
        f"{rounds},"
        f"{window},"
        f"{threshold},"
        f"{conf_low},"
        f"{conf_high},"
        f"{json_metadata}"
    )


CSV_HEADER = csv_line(
    rounds="r",
    window="w",
    threshold="th",
    conf_low="CI_l",
    conf_high="CI_h",
    json_metadata="json_metadata",
    is_header=True,
)