#!/usr/bin/env python3
"""Plot independent holdout predictions and relative-error distributions."""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=Path(__file__).resolve().parents[2] / 'results/model')
    args = parser.parse_args()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    bounds = []
    for mode, color in [('warm', '#2878b5'), ('cold', '#d45a25')]:
        table = pd.read_csv(args.results / f'v2-{mode}-holdout.csv')
        actual, predicted = table.measured_us.to_numpy(), table.prediction_us.to_numpy()
        axes[0].scatter(actual, predicted, s=13, alpha=.45, color=color, label=f'{mode}: 40 matrices / 480 candidates')
        bounds.extend([actual.min(), actual.max(), predicted.min(), predicted.max()])
        error = np.sort(np.abs(predicted / actual - 1) * 100)
        axes[1].plot(error, np.arange(1, len(error)+1) / len(error), color=color, label=mode)
    limits = [min(bounds)*.8, max(bounds)*1.2]
    axes[0].plot(limits, limits, '--', color='black', linewidth=1, label='prediction = measurement')
    axes[0].set(xscale='log', yscale='log', xlim=limits, ylim=limits,
                xlabel='Measured kernel time (µs)', ylabel='Predicted kernel time (µs)',
                title='Frozen model: independent holdout')
    axes[1].axvline(50, color='black', linestyle='--', linewidth=1)
    axes[1].axhline(.9, color='gray', linestyle=':', linewidth=1)
    axes[1].set(xlabel='Absolute relative error (%)', ylabel='Fraction of candidates',
                xlim=(0, 130), ylim=(0, 1.02), title='Error CDF (both pass the P90 ≤ 50% gate)')
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    for suffix in ['png', 'pdf']:
        fig.savefig(args.results / f'v2-validation.{suffix}', dpi=180)
    plt.close(fig)

if __name__ == '__main__':
    main()
