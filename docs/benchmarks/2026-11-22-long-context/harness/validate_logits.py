"""Check saved logits and compare them with the first Velri CPU F32 run at each length."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def validate(path, snapshot_dir, cpu_reference):
    record = json.loads(path.read_text())
    raw = (snapshot_dir / (path.stem + '.f32')).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == record['logits_sha256']
    logits = np.frombuffer(raw, '<f4').reshape(33, 8192)
    assert np.isfinite(logits).all()
    predictions = logits.argmax(axis=1)
    assert predictions.tolist() == record['greedy']

    with path.with_suffix('.csv').open() as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == len(logits)
    assert [int(row['token']) for row in rows] == record['greedy']
    times_us = [float(row['us']) for row in rows]
    assert abs(record['prefill_tps'] - record['prompt'] * 1e6 / times_us[0]) < 1e-7
    assert abs(record['decode_tps'] - 32e6 / sum(times_us[1:])) < 1e-7

    reference_path = cpu_reference / f"velri-cpu-f32-{record['prompt']}-1.f32"
    reference = np.fromfile(reference_path, '<f4').reshape(logits.shape)
    assert np.isfinite(reference).all()
    return {
        'engine': record['engine'],
        'prompt': record['prompt'],
        'block': record['block'],
        'finite_logits': logits.size,
        'top1_agreements_with_velri_cpu_f32_first_block': int((predictions == reference.argmax(axis=1)).sum()),
        'max_abs_diff_with_velri_cpu_f32_first_block': float(np.abs(logits - reference).max()),
        'logits_sha256': digest,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('snapshots', type=Path)
    parser.add_argument('--cpu-reference', type=Path, help='separate directory with CPU snapshots')
    args = parser.parse_args()
    records = sorted((args.output / 'records').glob('*.json'))
    checks = [validate(path, args.snapshots, args.cpu_reference or args.snapshots)
              for path in records if not path.name.endswith('.failure.json')]
    (args.output / 'numeric.json').write_text(json.dumps(checks, indent=2) + '\n')


if __name__ == '__main__':
    main()
