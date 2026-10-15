"""Collect aggregate research evidence; private route identities stay in the cache."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from decoder import decode
from experiment import get_windows, metrics, stat
from prepare import wrap


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--out', type=Path, default=Path('results'))
    args = p.parse_args()
    root = args.out
    definitions = json.loads((root / 'experiments.json').read_text())
    lock = json.loads((root / 'selection_lock.json').read_text())
    assert lock['variant_grid_sha256'] == hashlib.sha256((root / 'experiments.json').read_bytes()).hexdigest()
    result = {'protocol': json.loads((root / 'protocol.json').read_text()),
              'selection': lock, 'exact_search_probe': json.loads((root / 'exact_search_probe.json').read_text()),
              'variants': {}, 'execution': {}, 'model_summary': {}, 'prefix_analysis': {}, 'audits': {}}
    scenes, episodes = {}, {}
    for split in ('train', 'validation', 'test'):
        with np.load(args.cache / (split + '.npz')) as x:
            scenes[split], episodes[split] = set(x['scenes']), set(x['episode_ids'])
    for i, a in enumerate(scenes):
        for b in list(scenes)[i + 1:]:
            assert not scenes[a] & scenes[b] and not episodes[a] & episodes[b]
    result['audits']['scene_and_episode_overlap'] = 0
    for path in sorted((root / 'models').glob('*.npz')):
        with np.load(path) as model:
            config = json.loads(str(model['config']))
            training = json.loads(str(model['training'])) if 'training' in model else None
            result['model_summary'][path.stem] = {'config': config, 'training': training,
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}
    preferred = lock['selected_on_validation']['preferred_5point_3token_full_window']
    decoder_audit = {'model': definitions['variants'][preferred]['model'], 'windows': 0,
                     'max_xy_abs_difference_m': 0., 'max_yaw_difference_rad': 0.}
    for split in ('validation', 'test'):
        data = dict(np.load(args.cache / (split + '.npz')))
        n_routes = len(data['starts'])
        route_scenes = data['route_scenes']
        unique = np.unique(route_scenes)
        counts = np.bincount(route_scenes)
        rng = np.random.default_rng(2026101504)
        draws = rng.integers(len(unique), size=(2000, len(unique)))
        denom = counts[draws].sum(-1)
        baselines = {}
        for base in ('k256', 'official_release'):
            with np.load(root / split / base / 'private_predictions.npz') as x:
                baselines[base] = x['routes'][:, 0]
        for name, spec in definitions['variants'].items():
            s = json.loads((root / split / name / 'metrics.json').read_text())
            with np.load(root / split / name / 'private_predictions.npz') as pred:
                rows = pred['routes']
                assert np.array_equal(pred['route_scenes'], route_scenes) and len(rows) == n_routes
                sums = np.bincount(route_scenes, weights=rows[:, 0])
                boot = sums[draws].sum(-1) / denom
                s['route_endpoint_mean_scene_95ci'] = np.quantile(boot, [.025, .975]).tolist()
                s['paired'] = {}
                for base, values in baselines.items():
                    delta = rows[:, 0] - values
                    sums = np.bincount(route_scenes, weights=delta)
                    boot = sums[draws].sum(-1) / denom
                    s['paired'][base] = {'mean_delta_m': float(delta.mean()),
                        'scene_95ci_delta_m': np.quantile(boot, [.025, .975]).tolist(),
                        'lower_endpoint_fraction': float(np.mean(delta < 0))}
                if spec['sampling'] in ('fixed', 'quarter') and (spec['horizon'] >= (10 if spec['sampling'] == 'quarter' else 5)):
                    decoded = pred['decoded']
                    points = decoded[:, 1::2] if spec['sampling'] == 'quarter' else decoded[:, :5]
                    error = np.linalg.norm(points[..., :2] - data['fixed'][:, :5, :2], axis=-1)
                    valid = data['fixed_valid'][:, :5]
                    s['shared_first5_ade_m'] = stat((error * valid).sum(-1) / valid.sum(-1))
                if name == preferred:
                    decoded = pred['decoded']
                    independent = decode(root / 'models' / (spec['model'] + '.npz'), pred['codes'])
                    decoder_audit['windows'] += len(independent)
                    decoder_audit['max_xy_abs_difference_m'] = max(decoder_audit['max_xy_abs_difference_m'], float(np.max(abs(independent[..., :2] - decoded[..., :2]))))
                    decoder_audit['max_yaw_difference_rad'] = max(decoder_audit['max_yaw_difference_rad'], float(np.max(abs(wrap(independent[..., 2] - decoded[..., 2])))))
                    assert decoder_audit['max_xy_abs_difference_m'] < 2e-6 and decoder_audit['max_yaw_difference_rad'] < 2e-6
                    result['prefix_analysis'][split] = {}
                    target, valid, ai = get_windows(data, spec)
                    for level in (1, 2, 3):
                        reconstructed = decode(root / 'models' / (spec['model'] + '.npz'), pred['codes'][:, :level])
                        prefix, _, _ = metrics(data, reconstructed, target, valid, ai)
                        result['prefix_analysis'][split][str(level)] = prefix
                if name in ('official_release', 'k256', preferred, 'new_h1_diff_y1_beam'):
                    step = pred['steps']
                    error = wrap(step[:, 2] - data['truth'][:, 2]) * 180 / np.pi
                    truth = data['truth'][:, 2] * 180 / np.pi
                    pairs = np.ones(len(error) - 1, bool)
                    pairs[data['offsets'][1:-1] - 1] = False
                    s['yaw_diagnostics'] = {'lag1_residual_correlation': float(np.corrcoef(error[:-1][pairs], error[1:][pairs])[0, 1]) if error.std() > 1e-12 else 0,
                        'signed_error_by_true_turn': {label: stat(error[mask]) for label, mask in {
                            'right_ge5': truth <= -5, 'straight_lt1': abs(truth) < 1, 'left_ge5': truth >= 5}.items()},
                        'decoded_vs_true_slope': float(np.polyfit(truth, step[:, 2] * 180 / np.pi, 1)[0])}
            result['variants'].setdefault(name, {})[split] = s
        result['execution'][split] = json.loads((root / split / 'execution.json').read_text())
        expected_execution = {name for name, cfg in definitions['variants'].items() if cfg['kind'] in ('encode', 'release', 'factorized')}
        assert result['execution'][split].keys() == expected_execution
        with np.load(root / split / 'official_reproduce/private_predictions.npz') as reproduced, np.load(root / split / 'official_release/private_predictions.npz') as released:
            changed = np.flatnonzero(np.any(reproduced['codes'] != released['codes'], axis=-1))
            gaps = []
            for index in changed:
                target = data['original_target'][index]
                def score(pred):
                    return np.linalg.norm(pred[:, :2] - target[:, :2], axis=-1).mean() + .3 * abs(wrap(pred[-1, 2] - target[-1, 2]))
                gaps.append(float(abs(score(reproduced['decoded'][index]) - score(released['decoded'][index]))))
            result['audits'][split + '_encoder_near_ties'] = {'different_code_windows': len(changed),
                'max_objective_difference': max(gaps, default=0.),
                'mean_route_endpoint_delta_m': float(np.mean(reproduced['routes'][:, 0] - released['routes'][:, 0]))}
    result['audits']['independent_numpy_decoder'] = decoder_audit
    result['audits']['validation_encoder_reproduction'] = result['variants']['official_reproduce']['validation']['reproduction']
    result['audits']['test_encoder_reproduction'] = result['variants']['official_reproduce']['test']['reproduction']
    result['audits']['identity_max_endpoint_m'] = {split: result['variants']['identity'][split]['identity_max_endpoint_m'] for split in ('validation', 'test')}
    result['limitations'] = [
        'Offline quantization only: no learned policy, visual errors, collisions, or physical controller.',
        'All new RVQ codebooks fit train scenes only. Published K256 was fit on the full 779860-route cohort, so the new test split is not held out from its fitting cohort.',
        'Original spatially subdivided windows need GT-derived sidecar timestamps to locate the 0.5 s waypoint. Fixed/quarter contracts have a deterministic time base.',
        'Intervals resample scenes and retain route-count weighting; they do not estimate all codebook-training uncertainty or correct simultaneous comparisons.',
        'Original official codebook pretraining-distribution overlap is unknown. New books or re-encoded targets require matching decoder assets and policy relabeling/fine-tuning.',
        'The exact-search probe evaluates 64 validation windows, not global full-route optimality.',
        'Single-step-only losses deliberately sacrifice lookahead accuracy; their horizon label is not a guarantee of accurate future poses.'
    ]
    output = root / 'research_summary.json'
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(output, flush=True)


if __name__ == '__main__':
    main()
