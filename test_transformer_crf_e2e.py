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
    run(["-m", "transformer_crf.generate_fake_data"])
    run([
        "-m", "transformer_crf.train",
        "--vocab-file", "transformer_crf/demo_data/vocab.json",
        "--train-file", "transformer_crf/demo_data/train.json",
        "--train-format", "jsonl",
        "--dev-file", "transformer_crf/demo_data/dev.json",
        "--dev-format", "jsonl",
        "--model-dir", "transformer_crf/demo_data/model",
        "--epochs", "12",
    ])
    eval_out = run([
        "-m", "transformer_crf.evaluate",
        "--vocab-file", "transformer_crf/demo_data/vocab.json",
        "--model-dir", "transformer_crf/demo_data/model",
        "--eval-file", "transformer_crf/demo_data/dev.json",
        "--eval-format", "jsonl",
    ])
    if "Eval metrics:" not in eval_out:
        raise SystemExit("eval output missing")
    pred_out = run([
        "-m", "transformer_crf.predict",
        "--vocab-file", "transformer_crf/demo_data/vocab.json",
        "--model-path", "transformer_crf/demo_data/model/ner.pt",
        "--text", "高勇，男，中国国籍，无境外居留权",
    ])
    if "-->" not in pred_out:
        raise SystemExit("prediction output missing")
    model_path = os.path.join(ROOT, "transformer_crf/demo_data/model/ner.pt")
    if not os.path.exists(model_path):
        raise SystemExit("model checkpoint missing")
    print("ALL_CHECKS_PASSED")


if __name__ == "__main__":
    main()
