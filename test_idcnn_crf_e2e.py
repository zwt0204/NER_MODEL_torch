import os
import subprocess

ROOT = "/root/.openclaw/workspace/NER_MODEL_torch"
PYTHON = "/tmp/ner_torch_venv/bin/python"
ENV = dict(os.environ)
ENV["PYTHONPATH"] = ROOT
ENV.setdefault("OMP_NUM_THREADS", "1")
ENV.setdefault("MKL_NUM_THREADS", "1")
ENV.setdefault("OPENBLAS_NUM_THREADS", "1")
ENV.setdefault("NUMEXPR_NUM_THREADS", "1")
ENV.setdefault("TOKENIZERS_PARALLELISM", "false")


def run(args):
    cmd = [PYTHON] + args
    print("RUN", " ".join(cmd))
    result = subprocess.run(cmd, cwd=ROOT, env=ENV, check=True, text=True, capture_output=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result.stdout


def main():
    run(["-m", "IDCNN_CRF.generate_fake_data"])
    run([
        "-m", "IDCNN_CRF.train",
        "--vocab-file", "IDCNN_CRF/demo_data/vocab.json",
        "--train-file", "IDCNN_CRF/demo_data/train.json",
        "--train-format", "jsonl",
        "--dev-file", "IDCNN_CRF/demo_data/dev.json",
        "--dev-format", "jsonl",
        "--model-dir", "IDCNN_CRF/demo_data/model",
        "--epochs", "3",
    ])
    eval_out = run([
        "-m", "IDCNN_CRF.evaluate",
        "--vocab-file", "IDCNN_CRF/demo_data/vocab.json",
        "--model-dir", "IDCNN_CRF/demo_data/model",
        "--eval-file", "IDCNN_CRF/demo_data/dev.json",
        "--eval-format", "jsonl",
    ])
    if "Eval metrics:" not in eval_out:
        raise SystemExit("eval output missing")
    pred_out = run([
        "-m", "IDCNN_CRF.predict",
        "--vocab-file", "IDCNN_CRF/demo_data/vocab.json",
        "--model-path", "IDCNN_CRF/demo_data/model/ner.pt",
        "--text", "请问肯德基优惠券在哪里",
    ])
    if "-->" not in pred_out:
        raise SystemExit("prediction output missing")
    model_path = os.path.join(ROOT, "IDCNN_CRF/demo_data/model/ner.pt")
    if not os.path.exists(model_path):
        raise SystemExit("model checkpoint missing")
    print("ALL_CHECKS_PASSED")


if __name__ == "__main__":
    main()
