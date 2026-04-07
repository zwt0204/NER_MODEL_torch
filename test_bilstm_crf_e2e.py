import os
import subprocess

ROOT = "/root/.openclaw/workspace/NER_MODEL"
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


def write_conll(path):
    samples = [
        [("肯", "B-BRD"), ("德", "I-BRD"), ("基", "I-BRD"), ("优", "B-KWD"), ("惠", "I-KWD"), ("券", "I-KWD")],
        [("麦", "B-BRD"), ("当", "I-BRD"), ("劳", "I-BRD"), ("地", "B-KWD"), ("址", "I-KWD")],
    ]
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            for ch, tag in sample:
                f.write(f"{ch} {tag}\n")
            f.write("\n")


def main():
    run(["-m", "BILSTM_CRF.generate_fake_data"])
    run([
        "-m", "BILSTM_CRF.train",
        "--vocab-file", "BILSTM_CRF/demo_data/vocab.json",
        "--train-file", "BILSTM_CRF/demo_data/train.json",
        "--train-format", "jsonl",
        "--dev-file", "BILSTM_CRF/demo_data/dev.json",
        "--dev-format", "jsonl",
        "--model-dir", "BILSTM_CRF/demo_data/model",
        "--epochs", "3",
    ])
    eval_out = run([
        "-m", "BILSTM_CRF.evaluate",
        "--vocab-file", "BILSTM_CRF/demo_data/vocab.json",
        "--model-dir", "BILSTM_CRF/demo_data/model",
        "--eval-file", "BILSTM_CRF/demo_data/dev.json",
        "--eval-format", "jsonl",
    ])
    if "Eval metrics:" not in eval_out:
        raise SystemExit("eval output missing")
    pred_out = run([
        "-m", "BILSTM_CRF.predict",
        "--vocab-file", "BILSTM_CRF/demo_data/vocab.json",
        "--model-path", "BILSTM_CRF/demo_data/model/ner.pt",
        "--text", "请问肯德基优惠券在哪里",
    ])
    if "-->" not in pred_out:
        raise SystemExit("prediction output missing")

    conll_path = os.path.join(ROOT, "BILSTM_CRF/demo_data/sample.conll")
    write_conll(conll_path)
    run([
        "-m", "BILSTM_CRF.dataset_adapters",
        "--input", conll_path,
        "--output-dir", "BILSTM_CRF/demo_data/converted",
        "--format", "conll",
    ])

    model_path = os.path.join(ROOT, "BILSTM_CRF/demo_data/model/ner.pt")
    if not os.path.exists(model_path):
        raise SystemExit("model checkpoint missing")
    print("ALL_CHECKS_PASSED")


if __name__ == "__main__":
    main()
