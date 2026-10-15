"""NumPy-only decoder for the research RVQ codebooks (not a visual policy)."""
import json
from pathlib import Path
import numpy as np


def decode(path, codes):
    with np.load(Path(path)) as model:
        config = json.loads(str(model['config']))
        books = model['codebooks']
    codes = np.asarray(codes)
    if codes.ndim < 1 or codes.shape[-1] > len(books) or codes.shape[-1] < 1:
        raise ValueError('expected one to available-levels code indices on the last axis')
    if not np.issubdtype(codes.dtype, np.integer) or np.any((codes < 0) | (codes > 255)):
        raise ValueError('RVQ indices must be integers in 0..255')
    vector = np.zeros((*codes.shape[:-1], books.shape[-1]), np.float64)
    for level in range(codes.shape[-1]):
        vector += books[level, codes[..., level]]
    h = config['horizon']
    rep = config['representation']
    points = (vector / np.asarray(config['scale'])).reshape(*codes.shape[:-1], h, 4 if rep == 'sincos' else 3)
    if rep == 'sincos':
        points = np.stack([points[..., 0], points[..., 1], np.arctan2(points[..., 2], points[..., 3])], -1)
    elif rep == 'diff':
        yaw = points[..., 2].cumsum(-1)
        previous = yaw - points[..., 2]
        c, s = np.cos(previous), np.sin(previous)
        points = np.stack([(c * points[..., 0] - s * points[..., 1]).cumsum(-1),
                           (s * points[..., 0] + c * points[..., 1]).cumsum(-1), yaw], -1)
    elif rep != 'abs':
        raise ValueError('unsupported representation: ' + rep)
    points[..., 2] = (points[..., 2] + np.pi) % (2 * np.pi) - np.pi
    return points


def self_check():
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        p = Path(directory) / 'model.npz'
        config = {'horizon': 1, 'representation': 'diff', 'scale': [1, 1, 2]}
        books = np.zeros((2, 256, 3)); books[0, 1] = [1, 0, np.pi]
        books[1, 2] = [.2, 0, 0]
        np.savez(p, codebooks=books, config=np.array(json.dumps(config)))
        assert np.allclose(decode(p, [1, 2]), [[1.2, 0, np.pi / 2]])
        try:
            decode(p, [256, 2])
        except ValueError:
            pass
        else:
            raise AssertionError('out-of-range code accepted')


if __name__ == '__main__':
    self_check()
    print('NumPy decoder self-check passed')
