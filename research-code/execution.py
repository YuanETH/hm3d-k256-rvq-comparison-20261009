"""Compare receding and chunk execution on a common 0.5 s grid."""
import argparse
import json
from pathlib import Path
import numpy as np
from prepare import wrap
from experiment import stat

DEFAULT_NAMES = ['official_release', 'official_exec30', 'official_prefix5',
    'new_h10_diff_y1_beam', 'new_h5_diff_y1_beam', 'new_h5_abs_y1_beam',
    'new_h5_sincos_y1_beam', 'new_h5_4token', 'new_h3_diff_y1_beam',
    'new_h1_diff_y1_beam', 'new_h5_control_weighted', 'new_h5_action_only',
    'new_h10_quarter_y1_beam']


def grid_points(data, decoded, spec):
    n, h = decoded.shape[:2]
    if spec['sampling'] == 'original':
        times = data['original_times'][:, :h]
        valid = data['original_valid'][:, :h]
        available = np.floor(np.where(valid, times, 0).max(-1) * 2 + 1e-6).astype(int)
        grid = np.zeros((n, max(available), 3), np.float64)
        for j in range(grid.shape[1]):
            match = np.isclose(times, (j + 1) * .5, atol=1e-7) & valid
            covered = available > j
            if not np.all(match[covered].sum(-1) == 1):
                raise ValueError('chunk lacks an exact valid source-time waypoint')
            grid[covered, j] = decoded[np.flatnonzero(covered), match[covered].argmax(-1)]
    else:
        stride = 2 if spec['sampling'] == 'quarter' else 1
        valid = data['quarter_valid' if stride == 2 else 'fixed_valid'][:, :h]
        available = valid.sum(-1) // stride
        grid = decoded[:, stride - 1::stride].astype(np.float64)
    if np.any(available < 1):
        raise ValueError('chunk does not cover one source action')
    return grid, available


def rollout(start, grid, available, max_steps):
    path = np.empty((len(grid), 3), np.float64)
    i, calls = 0, 0
    base = start.copy()
    while i < len(grid):
        use = min(int(available[i]), max_steps, len(grid) - i)
        chunk = grid[i, :use]
        c, s = np.cos(base[2]), np.sin(base[2])
        path[i:i + use, 0] = base[0] + c * chunk[:, 0] - s * chunk[:, 1]
        path[i:i + use, 1] = base[1] + s * chunk[:, 0] + c * chunk[:, 1]
        path[i:i + use, 2] = wrap(base[2] + chunk[:, 2])
        base = path[i + use - 1].copy()
        i += use; calls += 1
    return path, calls


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--out', type=Path, default=Path('results'))
    p.add_argument('--split', choices=('validation', 'test'), default='validation')
    p.add_argument('--names')
    p.add_argument('--all', action='store_true')
    args = p.parse_args()
    if args.split == 'test' and not (args.out / 'selection_lock.json').exists():
        raise RuntimeError('test selection is not locked')
    data = dict(np.load(args.cache / (args.split + '.npz')))
    definitions = json.loads((args.out / 'experiments.json').read_text())['variants']
    names = args.names.split(',') if args.names else DEFAULT_NAMES
    if args.all:
        names = [name for name, spec in definitions.items() if spec['kind'] in ('encode', 'release', 'factorized')]
    output = args.out / args.split / 'execution.json'
    result = json.loads(output.read_text()) if output.exists() else {}
    for name in names:
        if name in result:
            continue
        spec = definitions[name]
        with np.load(args.out / args.split / name / 'private_predictions.npz') as x:
            decoded = x['decoded']
        grid, available = grid_points(data, decoded, spec)
        result[name] = {}
        for label, max_steps in [('step', 1), ('up_to_1s', 2), ('full_chunk', 100)]:
            rows, calls = [], 0
            for i, (lo, hi) in enumerate(zip(data['offsets'][:-1], data['offsets'][1:])):
                path, count = rollout(data['starts'][i], grid[lo:hi], available[lo:hi], max_steps)
                calls += count
                error = np.linalg.norm(path[:, :2] - data['gt_next'][lo:hi, :2], axis=-1)
                complete = np.vstack([data['starts'][i], path])
                speed = np.linalg.norm(np.diff(complete[:, :2], axis=0), axis=-1) / .5
                jerk = np.diff(speed, n=2) / .25
                yaw_acc = np.diff(wrap(np.diff(complete[:, 2]))) / .25 * 180 / np.pi
                rows.append([error[-1], error.mean(), abs(wrap(path[-1, 2] - data['gt_next'][hi - 1, 2])) * 180 / np.pi,
                             np.mean(abs(jerk)) if len(jerk) else np.nan,
                             np.mean(abs(yaw_acc)) if len(yaw_acc) else np.nan, count])
            rows = np.asarray(rows)
            summary = {'endpoint_m': stat(rows[:, 0]), 'ade_m': stat(rows[:, 1]),
                'heading_deg': stat(rows[:, 2]), 'speed_jerk_m_s3': stat(rows[np.isfinite(rows[:, 3]), 3]),
                'yaw_acc_deg_s2': stat(rows[np.isfinite(rows[:, 4]), 4]), 'requests': calls,
                'motion_tokens': calls * spec['tokens'], 'request_fraction_vs_k256': calls / len(grid),
                'mean_execution_s': len(grid) * .5 / calls}
            result[name][label] = summary
            np.savez(args.out / args.split / name / f'private_execution_{label}.npz', routes=rows)
        original = json.loads((args.out / args.split / name / 'metrics.json').read_text())
        assert abs(result[name]['step']['endpoint_m']['mean'] - original['route_endpoint_m']['mean']) < 1e-6
        print(name, {key: round(value['endpoint_m']['mean'], 5) for key, value in result[name].items()}, flush=True)
        output.write_text(json.dumps(result, indent=2) + '\n')
    output.write_text(json.dumps(result, indent=2) + '\n')


def self_check():
    start = np.array([0., 0., 0.])
    grid = np.array([[[1., 0., np.pi / 2], [1., 1., np.pi / 2]],
                     [[1., 0., 0.], [2., 0., 0.]]])
    path, calls = rollout(start, grid, np.array([2, 1]), 2)
    assert calls == 1 and np.allclose(path[-1], [1., 1., np.pi / 2])
    path, calls = rollout(start, grid, np.array([2, 1]), 1)
    assert calls == 2 and np.allclose(path[-1], [1., 1., np.pi / 2])


if __name__ == '__main__':
    self_check()
    main()
