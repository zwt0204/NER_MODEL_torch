# -*- encoding: utf-8 -*-
from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON = '/tmp/ner_torch_venv/bin/python'
ENV = os.environ.copy()
ENV['PYTHONPATH'] = ROOT + os.pathsep + ENV.get('PYTHONPATH', '')
ENV.setdefault('OMP_NUM_THREADS', '1')
ENV.setdefault('MKL_NUM_THREADS', '1')
ENV.setdefault('OPENBLAS_NUM_THREADS', '1')
ENV.setdefault('NUMEXPR_NUM_THREADS', '1')
ENV.setdefault('TOKENIZERS_PARALLELISM', 'false')

TESTS = [
    'test_albert_crf_ner_e2e.py',
    'test_albert_bisltm_crf_e2e.py',
]


def run(script_name: str):
    cmd = [PYTHON, script_name]
    print('RUN', ' '.join(cmd))
    result = subprocess.run(cmd, cwd=ROOT, env=ENV, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main():
    for script in TESTS:
        run(script)
    print('ALL_CHECKS_PASSED')


if __name__ == '__main__':
    main()
