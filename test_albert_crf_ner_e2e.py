import os
import subprocess

ROOT = "/root/.openclaw/workspace/NER_MODEL_torch"
PYTHON = "/tmp/ner_torch_venv/bin/python"
ENV = dict(os.environ)
ENV["PYTHONPATH"] = ROOT


def run(args):
    cmd = [PYTHON] + args
    print("RUN", " ".join(cmd))
    result = subprocess.run(cmd, cwd=ROOT, env=ENV, check=True, text=True, capture_output=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result.stdout


def main():
    run(["-m", "albert_crf_ner.generate_fake_data"])
    run([
        "-m", "albert_crf_ner.train",
        "--vocab-file", "albert_crf_ner/demo_data/vocab.json",
        "--train-file", "albert_crf_ner/demo_data/train.json",
        "--train-format", "jsonl",
        "--dev-file", "albert_crf_ner/demo_data/dev.json",
        "--dev-format", "jsonl",
        "--model-dir", "albert_crf_ner/demo_data/model",
        "--epochs", "10",
    ])
    eval_out = run([
        "-m", "albert_crf_ner.evaluate",
        "--vocab-file", "albert_crf_ner/demo_data/vocab.json",
        "--model-dir", "albert_crf_ner/demo_data/model",
        "--eval-file", "albert_crf_ner/demo_data/dev.json",
        "--eval-format", "jsonl",
    ])
    if "Eval metrics:" not in eval_out:
        raise SystemExit("eval output missing")
    pred_out = run([
        "-m", "albert_crf_ner.predict",
        "--vocab-file", "albert_crf_ner/demo_data/vocab.json",
        "--model-path", "albert_crf_ner/demo_data/model/ner.pt",
        "--text", "请问肯德基优惠券在哪里",
    ])
    if "-->" not in pred_out:
        raise SystemExit("prediction output missing")
    model_path = os.path.join(ROOT, "albert_crf_ner/demo_data/model/ner.pt")
    if not os.path.exists(model_path):
        raise SystemExit("model checkpoint missing")
    print("ALL_CHECKS_PASSED")


if __name__ == "__main__":
    main()
