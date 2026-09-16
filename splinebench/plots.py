def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_sample_efficiency(df, out, metric="pos_rmse", by=None, motions=None, noise_std=None):
    plt = _mpl()
    data = df
    if motions is not None:
        data = data[data["motion"].isin(motions)]
    if noise_std is not None:
        data = data[data["noise_std"] == noise_std]
    if by is None:
        by = [c for c in ["representation", "knots", "sampler" if "sampler" in data else "sampling"] if c in data.columns]
    fig, axes = plt.subplots(1, max(len(data["motion"].unique()), 1), figsize=(5.0 * max(len(data["motion"].unique()), 1), 4.0), squeeze=False)
    for ax, motion in zip(axes[0], sorted(data["motion"].unique())):
        sub = data[data["motion"] == motion]
        for key, g in sub.groupby(by[0]):
            curve = g.groupby("budget")[metric].mean()
            ax.plot(curve.index, curve.values, marker="o", label=str(key))
        ax.set_xscale("log")
        ax.set_xlabel("sample budget")
        ax.set_ylabel(metric)
        ax.set_title(motion)
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_pareto(df, out, x="n_params_total", y="pos_rmse", label=("representation", "knots"), motion=None):
    plt = _mpl()
    from .metrics import pareto_front

    data = df if motion is None else df[df["motion"] == motion]
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    for key, g in data.groupby(list(label)):
        ax.scatter(g[x], g[y], s=18, alpha=0.6)
        ax.annotate(str(key), (g[x].mean(), g[y].mean()), fontsize=6)
    front = pareto_front(data, x=x, y=y)
    front = front.sort_values(x)
    ax.plot(front[x], front[y], "k--", linewidth=1.0, label="pareto front")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_robustness(df, out, metric="pos_rmse"):
    plt = _mpl()
    data = df.copy()
    data["noise_label"] = data.apply(
        lambda r: f"std={r.get('noise_std', 0)}|out={r.get('outlier_frac', 0)}|miss={r.get('missing_frac', 0)}",
        axis=1,
    )
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for rep, g in data.groupby("representation"):
        curve = g.groupby("noise_label")[metric].mean()
        ax.plot(curve.index, curve.values, marker="o", label=str(rep))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7)
    ax.set_ylabel(metric)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out
