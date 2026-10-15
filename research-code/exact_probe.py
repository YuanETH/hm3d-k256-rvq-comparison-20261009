"""Exhaustive 256^3 code search on validation windows, no model refitting."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
import rvq
from experiment import get_windows, stat


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--official', type=Path, required=True)
    p.add_argument('--out', type=Path, default=Path('results'))
    p.add_argument('--device', default='cuda:7')
    args = p.parse_args()
    data = dict(np.load(args.cache / 'validation.npz'))
    pool = np.flatnonzero(data['fixed_valid'].all(-1))
    indices = np.random.default_rng(2026101503).choice(pool, 64, replace=False)
    definitions = json.loads((args.out / 'experiments.json').read_text())['variants']
    outputs = {'split': 'validation', 'windows': len(indices),
               'sample_indices_sha256': hashlib.sha256(indices.tobytes()).hexdigest(), 'models': {}}
    for name in ('official_reproduce', 'new_h5_abs_y1_beam'):
        spec = definitions[name]
        target, valid, ai = get_windows(data, spec)
        t, v, a = (torch.as_tensor(x[indices], device=args.device) for x in (target, valid, ai))
        if spec['family'] == 'official':
            books, cfg = rvq.official(args.official, spec['horizon'], args.device)
        else:
            books, cfg = rvq.load(args.out / 'models' / (spec['model'] + '.npz'), args.device)
            books = books[:3]
        pair = (books[0, :, None] + books[1, None]).flatten(0, 1)
        all_vectors = (pair[:, None] + books[2][None]).flatten(0, 1)
        all_decoded = rvq.decode(all_vectors, cfg)
        del pair, all_vectors
        with np.load(args.out / 'validation' / name / 'private_predictions.npz') as pred:
            approx = torch.as_tensor(pred['decoded'][indices], device=args.device)
        old_score = rvq.objective(approx[:, None], t, v, a, spec['objective']).squeeze(-1)
        best_values, best_codes = [], []
        start = time.perf_counter()
        # All tuples are visited; batches only bound temporary GPU memory.
        for i in range(len(indices)):
            scores = rvq.objective(all_decoded[None], t[i:i + 1], v[i:i + 1], a[i:i + 1], spec['objective']).squeeze(0)
            if 'stop_l0' in cfg:
                lo = cfg['stop_l0'] * 256 * 256
                scores[lo:lo + 256 * 256] = torch.inf
            value, flat = scores.min(0)
            best_values.append(value.item()); best_codes.append(int(flat))
            if (i + 1) % 8 == 0:
                print(name, i + 1, '/', len(indices), flush=True)
        best = np.asarray(best_values)
        old = old_score.cpu().numpy()
        gap = old - best
        assert gap.min() > -1e-5
        outputs['models'][name] = {'candidate_tuples_per_window': len(all_decoded) - (65536 if 'stop_l0' in cfg else 0),
            'approx_objective': stat(old), 'global_best_objective': stat(best),
            'objective_gap': stat(gap), 'within_1e_minus5_fraction': float(np.mean(gap <= 1e-5)),
            'mean_relative_objective_gap': float(np.mean(gap / np.maximum(best, 1e-8))),
            'enumeration_seconds': time.perf_counter() - start}
        del all_decoded
        torch.cuda.empty_cache()
    (args.out / 'exact_search_probe.json').write_text(json.dumps(outputs, indent=2) + '\n')


if __name__ == '__main__':
    main()
