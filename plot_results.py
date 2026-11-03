#!/usr/bin/python3
"""Summarise and plot NAGASIL runs.

Each result file (result/<graph>/<method>) holds one episode reward per line.
Reward = -log(believers / N), where believers = users in state 'I' (believing
the fake news) at endtime, so higher reward = fewer believers.
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

METHODS = ['random', 'maxinf', 'maxdef', 'gasil']
COLORS = {'random': '#8c8c8c', 'maxinf': '#4c78a8', 'maxdef': '#f58518', 'gasil': '#54a24b'}


def graph_size(graph):
    # graph dirs are named bG_<N>_<p>
    return int(graph.split('_')[1])


def load(resultroot):
    data = {}
    for graph in sorted(os.listdir(resultroot), key=lambda g: graph_size(g) if g.startswith('bG_') else 0):
        gdir = os.path.join(resultroot, graph)
        if not (os.path.isdir(gdir) and graph.startswith('bG_')):
            continue
        for m in METHODS:
            path = os.path.join(gdir, m)
            # result files are only flushed when a run finishes, so skip empty ones
            if os.path.exists(path) and os.path.getsize(path) > 0:
                r = np.loadtxt(path, ndmin=1)
                if r.size:
                    data.setdefault(graph, {})[m] = r
    return data


def smooth(x, w):
    w = max(1, min(w, len(x)))
    return np.convolve(x, np.ones(w) / w, mode='valid')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resultroot', default='result')
    parser.add_argument('--out', default='results/comparison.png')
    parser.add_argument('--last', type=int, default=20, help='episodes averaged for the "final" column')
    args = parser.parse_args()

    data = load(args.resultroot)
    if not data:
        raise SystemExit('no results found under ' + args.resultroot)

    rows = []
    header = '{:<11} {:<7} {:>4} {:>12} {:>12} {:>12} {:>16}'.format(
        'graph', 'method', 'eps', 'mean_reward', 'best_reward', 'final_rwd', 'final_believers')
    print(header)
    print('-' * len(header))
    for graph, runs in data.items():
        n = graph_size(graph)
        for m in METHODS:
            if m not in runs:
                continue
            r = runs[m]
            final = r[-args.last:].mean()
            believers = n * np.exp(-r[-args.last:]).mean()
            rows.append((graph, m, len(r), r.mean(), r.max(), final, believers))
            print('{:<11} {:<7} {:>4} {:>12.3f} {:>12.3f} {:>12.3f} {:>16.1f}'.format(*rows[-1]))

    graphs = list(data.keys())
    fig, axes = plt.subplots(2, len(graphs), figsize=(6 * len(graphs), 8), squeeze=False)
    for col, graph in enumerate(graphs):
        runs = data[graph]
        n = graph_size(graph)
        ms = [m for m in METHODS if m in runs]

        ax = axes[0][col]
        believers = [n * np.exp(-runs[m][-args.last:]).mean() for m in ms]
        bars = ax.bar(ms, believers, color=[COLORS[m] for m in ms])
        ax.bar_label(bars, fmt='%.1f')
        ax.set_title('{} (N={}): believers at end, last {} eps\n(lower is better)'.format(graph, n, args.last))
        ax.set_ylabel('# users believing fake news')

        ax = axes[1][col]
        for m in ms:
            w = 10
            ax.plot(np.arange(w, len(runs[m]) + 1), smooth(runs[m], w), label=m, color=COLORS[m])
        ax.set_title('{}: episode reward (moving avg, w=10)'.format(graph))
        ax.set_xlabel('iteration')
        ax.set_ylabel('reward = -log(believers/N)')
        ax.legend()

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print('\nsaved', args.out)


if __name__ == '__main__':
    main()
