import argparse
import os

from . import catalog, experiment, plots, suites


def _print_registries():
    for axis in ["representations", "knots", "fitters", "sampling", "time_param", "metrics"]:
        print(f"\n{axis}:")
        for name in catalog.CATALOG[axis]:
            status, cite = catalog.CATALOG[axis][name]
            print(f"  {status:>11}  {name:<32} {cite}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Spline-empirical experiment runner")
    parser.add_argument("--suite", default="smoke", choices=sorted(suites.SUITES))
    parser.add_argument("--full", action="store_true", help="use the full (non-quick) condition set")
    parser.add_argument("--out", default=None, help="JSONL output path")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--summary", default=None, help="write summary CSV")
    parser.add_argument("--plots", default=None, help="prefix for generated plots")
    parser.add_argument("--report", default=None, help="aggregate an existing JSONL file only")
    parser.add_argument("--list", action="store_true", help="list registry contents")
    args = parser.parse_args(argv)

    if args.list:
        _print_registries()
        return 0

    if args.report:
        df = experiment.load_results(args.report)
        summary = experiment.summarize(df, out_csv=args.summary)
        print(summary.to_string(index=False))
        if args.plots:
            os.makedirs(os.path.dirname(args.plots) or ".", exist_ok=True)
            plots.plot_sample_efficiency(df, f"{args.plots}_sample_efficiency.png")
            plots.plot_pareto(df, f"{args.plots}_pareto.png")
        return 0

    conditions = suites.build(args.suite, quick=not args.full)
    print(f"suite={args.suite} quick={not args.full} conditions={len(conditions)}")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    records = experiment.run_suite(conditions, out_path=args.out, limit=args.limit)
    print(f"completed {len(records)} conditions")
    if args.out and os.path.exists(args.out):
        df = experiment.load_results(args.out)
        if args.summary:
            summary = experiment.summarize(df, out_csv=args.summary)
            print(summary.to_string(index=False))
        if args.plots:
            os.makedirs(os.path.dirname(args.plots) or ".", exist_ok=True)
            plots.plot_sample_efficiency(df, f"{args.plots}_sample_efficiency.png")
            plots.plot_pareto(df, f"{args.plots}_pareto.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
