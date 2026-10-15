"""Run predeclared trajectory RVQ ablations without reading RGB or test scores."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
import rvq
from prepare import hashed, wrap

torch.set_num_threads(2)
torch.set_float32_matmul_precision('highest')


def definitions():
    models, variants = {}, {}
    variants['k256'] = {'family': 'baseline', 'kind': 'k256', 'tokens': 1, 'horizon': 1, 'sampling': 'fixed'}
    variants['identity'] = {'family': 'control', 'kind': 'identity', 'tokens': 0, 'horizon': 1, 'sampling': 'fixed'}
    variants['official_release'] = {'family': 'official', 'kind': 'release', 'tokens': 3, 'horizon': 10, 'sampling': 'original'}
    official = {
        'official_reproduce': ({'final_yaw': .3}, 10, 'original', False, 16, 0),
        'official_mask': ({'final_yaw': .3}, 10, 'original', True, 16, 0),
        'official_yaw_all03': ({'mean_yaw': .3}, 10, 'original', True, 16, 0),
        'official_yaw_all10': ({'mean_yaw': 1.}, 10, 'original', True, 16, 0),
        'official_exec03': ({'final_yaw': .3, 'action_yaw': .3}, 10, 'original', True, 16, 0),
        'official_exec10': ({'final_yaw': .3, 'action_yaw': 1.}, 10, 'original', True, 16, 0),
        'official_exec30': ({'final_yaw': .3, 'action_yaw': 3.}, 10, 'original', True, 16, 0),
        'official_prefix5': ({'mean_yaw': .3, 'action_yaw': 1.}, 5, 'original', True, 16, 0),
        'official_action_only': ({'ade': 0., 'action_xy': 1., 'action_yaw': 1.}, 10, 'original', True, 16, 1),
        'official_search64': ({'final_yaw': .3}, 10, 'original', True, 64, 1),
        'official_fixed05': ({'mean_yaw': .3, 'action_yaw': 1.}, 10, 'fixed', True, 16, 0),
        'official_fixed025': ({'mean_yaw': .3, 'action_yaw': 1.}, 10, 'quarter', True, 16, 0),
        'official_fixed05_action': ({'ade': 0., 'action_xy': 1., 'action_yaw': 1.}, 10, 'fixed', True, 16, 1),
        'official_fixed025_action': ({'ade': 0., 'action_xy': 1., 'action_yaw': 1.}, 10, 'quarter', True, 16, 1),
        'official_fixed05_prefix5': ({'mean_yaw': .3, 'action_yaw': 1.}, 5, 'fixed', True, 16, 0),
    }
    for name, (objective, horizon, sampling, mask, beam, refine) in official.items():
        variants[name] = {'family': 'official', 'kind': 'encode', 'tokens': 3, 'horizon': horizon,
                          'sampling': sampling, 'mask': mask, 'objective': objective, 'beam': beam, 'refine': refine}
    for h in (10, 5):
        for yaw in (.3, 1., 3.):
            name = f'new_h{h}_diff_y{yaw:g}'
            models[name] = {'horizon': h, 'sampling': 'fixed', 'representation': 'diff', 'yaw_scale': yaw,
                            'levels': 4 if h == 5 and yaw == 1 else 3, 'seed': 20261015}
    models.update({
        'new_h3_diff_y1': {'horizon': 3, 'sampling': 'fixed', 'representation': 'diff', 'yaw_scale': 1., 'levels': 3, 'seed': 20261015},
        'new_h1_diff_y1': {'horizon': 1, 'sampling': 'fixed', 'representation': 'diff', 'yaw_scale': 1., 'levels': 3, 'seed': 20261015},
        'new_h5_abs_y1': {'horizon': 5, 'sampling': 'fixed', 'representation': 'abs', 'yaw_scale': 1., 'levels': 3, 'seed': 20261015},
        'new_h5_sincos_y1': {'horizon': 5, 'sampling': 'fixed', 'representation': 'sincos', 'yaw_scale': 1., 'levels': 3, 'seed': 20261015},
        'new_h10_quarter_y1': {'horizon': 10, 'sampling': 'quarter', 'representation': 'diff', 'yaw_scale': 1., 'levels': 3, 'seed': 20261015},
        'new_original_y1': {'horizon': 10, 'sampling': 'original', 'representation': 'diff', 'yaw_scale': 1., 'levels': 3, 'seed': 20261015},
        'new_h5_balanced_y1': {'horizon': 5, 'sampling': 'fixed', 'representation': 'diff', 'yaw_scale': 1., 'levels': 3, 'seed': 20261015, 'balanced': True},
        'new_h5_diff_y1_seed2': {'horizon': 5, 'sampling': 'fixed', 'representation': 'diff', 'yaw_scale': 1., 'levels': 3, 'seed': 20261016},
        'new_h5_diff_y1_seed3': {'horizon': 5, 'sampling': 'fixed', 'representation': 'diff', 'yaw_scale': 1., 'levels': 3, 'seed': 20261017},
    })
    for name, model in models.items():
        for encoder in ('l2', 'beam'):
            variants[name + '_' + encoder] = {'family': 'new_codebook', 'kind': 'encode', 'model': name,
                'horizon': model['horizon'], 'sampling': model['sampling'], 'tokens': 3, 'levels': 3,
                'encoder': encoder, 'mask': True, 'beam': 8,
                'objective': {'mean_yaw': .3, 'action_yaw': 1.}}
    for levels in (1, 2):
        variants[f'new_h1_{levels}token'] = {**variants['new_h1_diff_y1_l2'], 'tokens': levels, 'levels': levels}
    variants['new_h5_4token'] = {**variants['new_h5_diff_y1_beam'], 'tokens': 4, 'levels': 4}
    for h in (3, 5, 10):
        variants[f'new_h{h}_action_only'] = {**variants[f'new_h{h}_diff_y1_beam'],
            'objective': {'ade': 0., 'action_xy': 1., 'action_yaw': 1.}, 'refine': 1, 'beam': 16}
    variants['new_h5_control_weighted'] = {**variants['new_h5_diff_y1_beam'],
        'objective': {'action_xy': 1., 'mean_yaw': .3, 'action_yaw': 1.}, 'beam': 16}
    for seed in (2, 3):
        variants[f'new_h5_control_seed{seed}'] = {**variants['new_h5_control_weighted'], 'model': f'new_h5_diff_y1_seed{seed}'}
    # Extra angle symbols are explicitly charged, and are not a pure three-token RVQ.
    variants['official_plus_angle8'] = {'family': 'hybrid', 'kind': 'angle', 'tokens': 4,
        'horizon': 10, 'sampling': 'original', 'angle_bits': 8, 'range_deg': 30.}
    models['new_h5_factorized'] = {'horizon': 5, 'sampling': 'fixed', 'representation': 'factorized',
                                   'yaw_scale': 1., 'levels': 2, 'seed': 20261015}
    for xy_levels, yaw_levels in ((2, 1), (1, 2)):
        variants[f'factorized_xy{xy_levels}_yaw{yaw_levels}'] = {
            'family': 'factorized', 'kind': 'factorized', 'model': 'new_h5_factorized',
            'horizon': 5, 'sampling': 'fixed', 'tokens': 3,
            'xy_levels': xy_levels, 'yaw_levels': yaw_levels}
    return models, variants


def get_windows(data, spec):
    h, sampling = spec['horizon'], spec['sampling']
    prefix = {'fixed': 'fixed', 'quarter': 'quarter', 'original': 'original_target'}[sampling]
    target = data[prefix][:, :h].copy()
    valid = data[{'fixed': 'fixed_valid', 'quarter': 'quarter_valid', 'original': 'original_valid'}[sampling]][:, :h].copy()
    action_index = data['action_index'].copy() if sampling == 'original' else np.full(len(target), 1 if sampling == 'quarter' else 0)
    if np.any(action_index >= h):
        raise ValueError('horizon lacks the 0.5 s execution point')
    if spec.get('mask') is False:
        valid[:] = True
    return target, valid, action_index.astype(np.int64)


def subset(data, per_scene):
    chosen = []
    for scene in np.unique(data['route_scenes']):
        ids = np.flatnonzero(data['route_scenes'] == scene)
        chosen.extend(sorted(ids, key=lambda i: hashed(str(data['episode_ids'][i])))[:per_scene])
    chosen = sorted(chosen)
    indices = np.concatenate([np.arange(data['offsets'][i], data['offsets'][i + 1]) for i in chosen])
    out = {k: v[indices] for k, v in data.items() if len(v.shape) and len(v) == len(data['truth'])}
    for key in ('starts', 'episode_ids', 'route_scenes'):
        out[key] = data[key][chosen]
    out['offsets'] = np.r_[0, np.cumsum([data['offsets'][i + 1] - data['offsets'][i] for i in chosen])]
    out['scenes'] = data['scenes']
    return out


def stat(values):
    x = np.asarray(values, np.float64)
    return {'n': len(x), 'mean': float(x.mean()), 'median': float(np.median(x)),
            'p95': float(np.quantile(x, .95)), 'p99': float(np.quantile(x, .99))}


def integrate(start, steps):
    previous = start[2] + np.r_[0., np.cumsum(steps[:-1, 2])]
    c, s = np.cos(previous), np.sin(previous)
    xy = np.column_stack([c * steps[:, 0] - s * steps[:, 1], s * steps[:, 0] + c * steps[:, 1]])
    return np.column_stack([start[:2] + xy.cumsum(0), wrap(start[2] + steps[:, 2].cumsum())])


def metrics(data, decoded, target, valid, action_index):
    step = decoded[np.arange(len(decoded)), action_index].astype(np.float64)
    truth = data['truth']
    xy = np.linalg.norm(step[:, :2] - truth[:, :2], axis=-1)
    yaw_signed = wrap(step[:, 2] - truth[:, 2]) * 180 / np.pi
    yaw = np.abs(yaw_signed)
    local_xy = np.linalg.norm(decoded[..., :2] - target[..., :2], axis=-1)
    local_yaw = np.abs(wrap(decoded[..., 2] - target[..., 2])) * 180 / np.pi
    local_ade = (local_xy * valid).sum(-1) / valid.sum(-1)
    travel = np.linalg.norm(np.diff(np.concatenate([np.zeros((len(target), 1, 2)), target[..., :2]], 1), axis=1), axis=-1)
    window_travel = (travel * valid).sum(-1)
    routes, segment5, segment10, identity_errors = [], [], [], []
    for i, (lo, hi) in enumerate(zip(data['offsets'][:-1], data['offsets'][1:])):
        gt = data['gt_next'][lo:hi]
        pred = integrate(data['starts'][i], step[lo:hi])
        error = np.linalg.norm(pred[:, :2] - gt[:, :2], axis=-1)
        identity = integrate(data['starts'][i], truth[lo:hi])
        identity_errors.append(float(np.linalg.norm(identity[-1, :2] - gt[-1, :2])))
        routes.append([error[-1], error.mean(), abs(wrap(pred[-1, 2] - gt[-1, 2])) * 180 / np.pi,
                       np.linalg.norm(truth[lo:hi, :2], axis=-1).sum(), hi - lo])
        for n, destination in ((10, segment5), (20, segment10)):
            for j in range(lo, hi - n + 1, n):
                start = data['starts'][i] if j == lo else data['gt_next'][j - 1]
                end = integrate(start, step[j:j + n])[-1]
                destination.append(np.linalg.norm(end[:2] - data['gt_next'][j + n - 1, :2]))
    routes = np.asarray(routes)
    result = {'route_endpoint_m': stat(routes[:, 0]), 'route_ade_m': stat(routes[:, 1]),
              'route_heading_deg': stat(routes[:, 2]), 'route_length_m': stat(routes[:, 3]),
              'step_xy_m': stat(xy), 'step_yaw_deg': stat(yaw), 'step_yaw_signed_deg': stat(yaw_signed),
              'local_ade_m': stat(local_ade), 'local_yaw_deg': stat(local_yaw[valid]),
              'window_travel_m': stat(window_travel), 'segment5_endpoint_m': stat(segment5),
              'segment10_endpoint_m': stat(segment10), 'identity_max_endpoint_m': max(identity_errors)}
    for label, mask in {
        'turn_lt1': abs(truth[:, 2]) < np.deg2rad(1),
        'turn_1to5': (abs(truth[:, 2]) >= np.deg2rad(1)) & (abs(truth[:, 2]) < np.deg2rad(5)),
        'turn_5to15': (abs(truth[:, 2]) >= np.deg2rad(5)) & (abs(truth[:, 2]) < np.deg2rad(15)),
        'turn_ge15': abs(truth[:, 2]) >= np.deg2rad(15),
        'in_place_turn': (np.linalg.norm(truth[:, :2], axis=-1) < .02) & (abs(truth[:, 2]) >= np.deg2rad(5)),
        'padded_window': ~valid.all(-1),
    }.items():
        if mask.any():
            result[label] = {'yaw_deg': stat(yaw[mask]), 'xy_m': stat(xy[mask])}
    assert result['identity_max_endpoint_m'] < 1e-7, result['identity_max_endpoint_m']
    return result, routes, step


def training(args, name, model):
    path = args.out / 'models' / (name + '.npz')
    if path.exists():
        print('cached model', name, flush=True); return
    data = dict(np.load(args.cache / 'train.npz'))
    available = data['fixed_valid'].all(-1)
    indices = []
    for scene in np.unique(data['action_scenes']):
        pool = np.flatnonzero(available & (data['action_scenes'] == scene))
        rng = np.random.default_rng(20261015 + int(scene))
        p = None
        if model.get('balanced'):
            bins = np.digitize(np.abs(data['truth'][pool, 2]) * 180 / np.pi, [1, 5, 15])
            counts = np.bincount(bins, minlength=4)
            p = 1 / counts[bins].clip(1); p = p / p.sum()
        indices.extend(rng.choice(pool, min(2048, len(pool)), replace=False, p=p))
    indices = np.asarray(indices)
    target, _, _ = get_windows(data, model)
    target = torch.as_tensor(target[indices], device=args.device)
    h, rep, yaw = model['horizon'], model['representation'], model['yaw_scale']
    scale = ([1., 1., yaw, yaw] if rep == 'sincos' else [1., 1., yaw]) * h
    config = {**model, 'scale': scale, 'training_windows': len(indices),
              'training_indices_sha256': hashlib.sha256(indices.tobytes()).hexdigest(),
              'training_rule': 'same complete 5-second anchors, max 2048 per train scene; no validation/test fitting'}
    if rep == 'factorized':
        payload = {}
        for key, vector in {'xy': target[..., :2].flatten(1), 'yaw': rvq.to_diff(target)[..., 2]}.items():
            residual, books = vector.clone(), []
            for level in range(2):
                trials = [rvq.kmeans(residual, iterations=args.iterations, seed=model['seed'] + level * 1009 + r)
                          for r in range(2)]
                cb, _ = min(trials, key=lambda z: z[1]['inertia_per_window'])
                ids = torch.cat([rvq.distances(b, cb).argmin(-1) for b in residual.split(32768)])
                residual -= cb[ids]; books.append(cb.cpu().numpy())
            payload[key] = np.stack(books)
        np.savez(path, **payload, config=np.array(json.dumps(config)))
    else:
        rvq.fit(target, config, path, levels=model['levels'], iterations=args.iterations, restarts=2, seed=model['seed'])


def evaluation(args, name, spec):
    folder = args.out / ('pilot' if args.pilot else args.split) / name
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'metrics.json').exists() and not args.force:
        print('cached evaluation', name, flush=True); return
    if args.split == 'test' and not (args.out / 'selection_lock.json').exists():
        raise RuntimeError('test is locked until validation selection is recorded')
    if args.split == 'test':
        lock = json.loads((args.out / 'selection_lock.json').read_text())
        if lock['variants'].get(name) != spec:
            raise RuntimeError('test variant differs from the frozen configuration')
        if spec.get('model'):
            actual = hashlib.sha256((args.out / 'models' / (spec['model'] + '.npz')).read_bytes()).hexdigest()
            if actual != lock['model_sha256'][spec['model']]:
                raise RuntimeError('test model differs from the frozen codebook')
    data = dict(np.load(args.cache / (args.split + '.npz')))
    if args.pilot:
        data = subset(data, 20)
    target, valid, ai = get_windows(data, spec)
    start = time.perf_counter()
    codes = None
    model_hash = None
    if spec['kind'] == 'k256':
        decoded = data['ksteps'][:, None].astype(np.float32)
    elif spec['kind'] == 'identity':
        decoded = data['truth'][:, None].copy()
    elif spec['kind'] in ('release', 'angle'):
        decoded = data['original_decoded'].copy()
        codes = data['original_codes'].copy()
        if spec['kind'] == 'angle':
            rows = np.arange(len(decoded))
            error = wrap(data['truth'][:, 2] - decoded[rows, ai, 2])
            limit = np.deg2rad(spec['range_deg'])
            q = np.rint(np.clip(error / limit, -1, 1) * 127).astype(int)
            decoded[rows, ai, 2] = wrap(decoded[rows, ai, 2] + q * limit / 127)
            # Only the executed 0.5 s heading is corrected; later lookahead is unchanged.
            spec = {**spec, 'angle_saturation_count': int(np.sum(abs(error) > limit))}
    elif spec['kind'] == 'factorized':
        model_path = args.out / 'models' / (spec['model'] + '.npz')
        model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
        t = torch.as_tensor(target, device=args.device)
        outputs, paths = {}, []
        with np.load(model_path) as model:
            for key, vector in {'xy': t[..., :2].flatten(1), 'yaw': rvq.to_diff(t)[..., 2]}.items():
                residual, recon = vector.clone(), torch.zeros_like(vector)
                for book in torch.as_tensor(model[key][:spec[key + '_levels']], device=args.device):
                    ids = torch.cat([rvq.distances(b, book).argmin(-1) for b in residual.split(16384)])
                    residual -= book[ids]; recon += book[ids]; paths.append(ids.cpu().numpy())
                outputs[key] = recon.cpu().numpy()
        decoded = np.concatenate([outputs['xy'].reshape(-1, spec['horizon'], 2),
                                  wrap(outputs['yaw'].cumsum(-1))[..., None]], -1)
        codes = np.stack(paths, -1).astype(np.int16)
    else:
        if spec['family'] == 'official':
            books, cfg = rvq.official(args.official, spec['horizon'], args.device)
        else:
            model_path = args.out / 'models' / (spec['model'] + '.npz')
            model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
            books, cfg = rvq.load(model_path, args.device, spec.get('levels', 3))
        t = torch.as_tensor(target, device=args.device)
        if spec.get('encoder') == 'l2':
            decoded, codes = rvq.encode_l2(t, books, cfg)
        else:
            fallback = torch.as_tensor(data['original_codes'].astype(np.int64), device=args.device) if spec['family'] == 'official' else None
            decoded, codes = rvq.encode(t, torch.as_tensor(valid, device=args.device), torch.as_tensor(ai, device=args.device),
                books, cfg, spec['objective'], beam_width=spec.get('beam', 16), batch_size=args.batch_size,
                fallback_codes=fallback, refine=spec.get('refine', 0))
    runtime = time.perf_counter() - start
    metric_valid = get_windows(data, {**spec, 'mask': True})[1]
    summary, routes, steps = metrics(data, decoded, target, metric_valid, ai)
    summary.update(variant=name, config=spec, split=args.split, pilot=args.pilot,
                   scenes=len(np.unique(data['route_scenes'])), routes=len(routes), steps=len(steps),
                   model_sha256=model_hash, encoding_wall_seconds=runtime)
    if name == 'official_reproduce':
        old = data['original_decoded']
        t = torch.as_tensor(target, device=args.device)
        v = torch.as_tensor(valid, device=args.device)
        a = torch.as_tensor(ai, device=args.device)
        before = rvq.objective(torch.as_tensor(old[:, None], device=args.device), t, v, a, spec['objective'])
        after = rvq.objective(torch.as_tensor(decoded[:, None], device=args.device), t, v, a, spec['objective'])
        summary['reproduction'] = {'max_objective_regression': float((after - before).max()),
            'same_code_tuple_fraction': float(np.mean(np.all(codes == data['original_codes'], axis=-1))),
            'decoded_max_difference': float(np.max(abs(old - decoded)))}
        assert summary['reproduction']['max_objective_regression'] < 1e-5
    np.savez(folder / 'private_predictions.npz', decoded=decoded, codes=np.array([]) if codes is None else codes,
             steps=steps, routes=routes, route_scenes=data['route_scenes'])
    (folder / 'metrics.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({'variant': name, 'steps': len(steps), 'encoding_s': runtime,
        'endpoint': summary['route_endpoint_m']['mean'], 'yaw': summary['step_yaw_deg']['mean'],
        'xy': summary['step_xy_m']['mean']}), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('define', 'train', 'evaluate'))
    p.add_argument('--name')
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--out', type=Path, default=Path('results'))
    p.add_argument('--official', type=Path)
    p.add_argument('--split', default='validation', choices=('validation', 'test'))
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--iterations', type=int, default=35)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--pilot', action='store_true')
    p.add_argument('--force', action='store_true')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'models').mkdir(exist_ok=True)
    models, variants = definitions()
    if args.action == 'define':
        (args.out / 'experiments.json').write_text(json.dumps({'models': models, 'variants': variants}, indent=2) + '\n')
        print(len(models), 'models', len(variants), 'variants')
    elif args.action == 'train':
        training(args, args.name, models[args.name])
    else:
        evaluation(args, args.name, variants[args.name])


if __name__ == '__main__':
    main()
