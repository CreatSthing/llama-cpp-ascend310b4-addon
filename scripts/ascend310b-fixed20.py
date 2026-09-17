#!/usr/bin/env python3
import argparse
import json
import pathlib
import re
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["fp16", "candidate"], required=True)
    args = parser.parse_args()

    questions = json.loads(pathlib.Path(args.questions).read_text(encoding="utf-8"))
    prompt = ["Answer all questions. Output exactly 20 lines in the form 1:B. Do not explain."]
    prompt.extend(f"{item['id']}. {item['question']}" for item in questions)
    prompt_text = "\n".join(prompt)
    command = [args.cli, "-m", args.model, "-ngl", "99", "-b", "1", "-ub", "1",
               "-n", "100", prompt_text]
    completed = subprocess.run(command, text=True, capture_output=True, timeout=300)
    raw = completed.stdout
    found = {int(i): answer for i, answer in re.findall(r"(?m)^\s*(\d{1,2})\s*[:.)-]\s*([A-D])\b", raw.upper())}
    results = []
    for item in questions:
        predicted = found.get(item["id"], "")
        results.append({**item, "predicted": predicted, "correct": predicted == item["answer"]})
    score = sum(x["correct"] for x in results)
    report = {"mode": args.mode, "score": score, "total": len(results),
              "raw": raw, "stderr": completed.stderr, "returncode": completed.returncode,
              "results": results}
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{args.mode}: {score}/{len(results)}")


if __name__ == "__main__":
    main()
