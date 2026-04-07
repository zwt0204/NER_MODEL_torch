# -*- encoding: utf-8 -*-
from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON = '/tmp/ner_torch_venv/bin/python'
ENV = os.environ.copy()
ENV['PYTHONPATH'] = ROOT + os.pathsep + ENV.get('PYTHONPATH', '')


def run(args):
    cmd = [PYTHON] + args
    print('RUN', ' '.join(cmd))
    result = subprocess.run(cmd, cwd=ROOT, env=ENV, check=True, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result.stdout


def main():
    run(['-m', 'albert_bisltm_crf.generate_fake_data'])
    run([
        '-m', 'albert_bisltm_crf.train',
        '--vocab-file', 'albert_bisltm_crf/demo_data/vocab.json',
        '--train-file', 'albert_bisltm_crf/demo_data/train.json',
        '--train-format', 'jsonl',
        '--dev-file', 'albert_bisltm_crf/demo_data/dev.json',
        '--dev-format', 'jsonl',
        '--model-dir', 'albert_bisltm_crf/demo_data/model',
        '--epochs', '8',
    ])
    eval_out = run([
        '-m', 'albert_bisltm_crf.evaluate',
        '--vocab-file', 'albert_bisltm_crf/demo_data/vocab.json',
        '--model-dir', 'albert_bisltm_crf/demo_data/model',
        '--eval-file', 'albert_bisltm_crf/demo_data/dev.json',
        '--eval-format', 'jsonl',
    ])
    if 'Eval metrics:' not in eval_out:
        raise SystemExit('eval output missing')
    pred_out = run([
        '-m', 'albert_bisltm_crf.predict',
        '--vocab-file', 'albert_bisltm_crf/demo_data/vocab.json',
        '--model-path', 'albert_bisltm_crf/demo_data/model/ner.pt',
        '--text', '请问肯德基优惠券在哪里',
    ])
    if '-->' not in pred_out:
        raise SystemExit('prediction output missing')
    model_path = os.path.join(ROOT, 'albert_bisltm_crf/demo_data/model/ner.pt')
    if not os.path.exists(model_path):
        raise SystemExit(f'model file missing: {model_path}')
    print('ALL_CHECKS_PASSED')


if __name__ == '__main__':
    main()
