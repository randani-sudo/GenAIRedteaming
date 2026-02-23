#!/usr/bin/env python3
"""
build_multilingual_dataset.py
Merges open-source jailbreak datasets + translates attacks via Helsinki-NLP MarianMT (GPU).
Usage:
  python3 build_multilingual_dataset.py
  python3 build_multilingual_dataset.py --translate --langs hi,ar,fr,es
"""

import argparse
import json
import os
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
import torch

LANG_MODEL_MAP = {
    "fr": "Helsinki-NLP/opus-mt-en-fr",
    "es": "Helsinki-NLP/opus-mt-en-es",
    "de": "Helsinki-NLP/opus-mt-en-de",
    "ar": "Helsinki-NLP/opus-mt-en-ar",
    "hi": "Helsinki-NLP/opus-mt-en-hi",
    "zh": "Helsinki-NLP/opus-mt-en-zh",
    "pt": "Helsinki-NLP/opus-mt-en-ROMANCE",
}

# ── Translator (GPU-first, CPU fallback) ─────────────────────────
class Translator:
    def __init__(self):
        self._cache = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("  Translator device:", self.device)

    def translate(self, texts, lang):
        if lang not in self._cache:
            print("  Loading Helsinki-NLP model for {}...".format(lang))
            from transformers import MarianMTModel, MarianTokenizer
            tok   = MarianTokenizer.from_pretrained(LANG_MODEL_MAP[lang])
            model = MarianMTModel.from_pretrained(LANG_MODEL_MAP[lang]).to(self.device)
            self._cache[lang] = (tok, model)
        tok, model = self._cache[lang]
        batch_size = 64 if self.device.type == "cuda" else 16
        out = []
        for i in range(0, len(texts), batch_size):
            batch  = texts[i:i+batch_size]
            inputs = tok(batch, return_tensors="pt", padding=True,
                         truncation=True, max_length=256)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                ids = model.generate(**inputs, max_length=256)
            out.extend(tok.batch_decode(ids, skip_special_tokens=True))
        return out


# ── Loaders ──────────────────────────────────────────────────────
def load_base_jsonl(path):
    rows = []
    if not os.path.exists(path):
        print("  NOT FOUND:", path)
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                rows.append({
                    "prompt" : str(obj.get("prompt", "")).strip(),
                    "label"  : int(obj.get("label", 0)),
                    "lang"   : obj.get("lang", "en"),
                    "source" : obj.get("source", "combined_dataset"),
                })
            except Exception:
                continue
    print("  {} rows".format(len(rows)))
    return rows


def load_rubend18():
    print("Loading rubend18/ChatGPT-Jailbreak-Prompts...")
    rows = []
    try:
        from datasets import load_dataset
        ds = load_dataset("rubend18/ChatGPT-Jailbreak-Prompts", split="train")
        for item in ds:
            text = str(item.get("Prompt") or item.get("prompt") or "").strip()
            if len(text) > 5:
                rows.append({
                    "prompt" : text,
                    "label"  : 1,
                    "lang"   : "en",
                    "source" : "rubend18/ChatGPT-Jailbreak-Prompts",
                })
        print("  {} rows".format(len(rows)))
    except Exception as e:
        print("  SKIP:", e)
    return rows


def load_trustairlab():
    print("Loading TrustAIRLab/in-the-wild-jailbreak-prompts...")
    rows = []
    configs = [
        ("jailbreak_2023_12_25", 1),
        ("jailbreak_2023_05_07", 1),
        ("regular_2023_12_25",   0),
        ("regular_2023_05_07",   0),
    ]
    try:
        from datasets import load_dataset
        for config_name, label in configs:
            try:
                ds = load_dataset("TrustAIRLab/in-the-wild-jailbreak-prompts",
                                  config_name, split="train")
                before = len(rows)
                for item in ds:
                    text = str(item.get("prompt") or item.get("text") or
                               item.get("content") or "").strip()
                    if len(text) > 5:
                        rows.append({
                            "prompt" : text,
                            "label"  : label,
                            "lang"   : "en",
                            "source" : "TrustAIRLab/" + config_name,
                        })
                print("  config={} label={} +{} rows".format(
                    config_name, label, len(rows) - before))
            except Exception as e:
                print("  config={} SKIP: {}".format(config_name, e))
    except Exception as e:
        print("  SKIP:", e)
    print("  {} total".format(len(rows)))
    return rows


def load_jailbreakbench():
    print("Loading JailbreakBench...")
    rows = []
    try:
        import jailbreakbench as jbb
        artifact = jbb.read_artifact(method="PAIR", model_name="vicuna-13b-v1.5")
        for jb in artifact.jailbreaks:
            prompt = (getattr(jb, "prompt", None) or
                      getattr(jb, "goal",   None) or
                      getattr(jb, "behavior", ""))
            if prompt and len(str(prompt).strip()) > 5:
                rows.append({
                    "prompt" : str(prompt).strip(),
                    "label"  : 1,
                    "lang"   : "en",
                    "source" : "JailbreakBench",
                })
        print("  {} rows".format(len(rows)))
    except Exception as e:
        print("  SKIP:", e)
    return rows


# ── Dedup ─────────────────────────────────────────────────────────
def deduplicate(rows):
    seen, out = set(), []
    for r in rows:
        key = r["prompt"].strip().lower()[:200]
        if key not in seen and len(r["prompt"].strip()) > 5:
            seen.add(key)
            out.append(r)
    return out


# ── Translation ───────────────────────────────────────────────────
def translate_attacks(rows, langs, max_per_lang=500):
    translator  = Translator()
    attack_rows = [r for r in rows if r["label"] == 1 and r["lang"] == "en"]
    sample      = attack_rows[:max_per_lang]
    texts       = [r["prompt"] for r in sample]
    new_rows    = []
    print("  Translating {} attack prompts per language".format(len(texts)))
    for lang in langs:
        print("  → {}".format(lang))
        try:
            translated = translator.translate(texts, lang)
            for orig, trans in zip(sample, translated):
                new_rows.append({
                    "prompt" : trans,
                    "label"  : 1,
                    "lang"   : lang,
                    "source" : orig["source"] + "__" + lang,
                })
            print("    {} rows added".format(len(translated)))
        except Exception as e:
            print("    FAILED {}: {}".format(lang, e))
    return new_rows


# ── Main ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-jsonl",
        default="/home/rohitandani/orchids-projects/gold-rhinoceros/ml-service/combined_dataset.jsonl")
    parser.add_argument("--output",
        default="/home/rohitandani/orchids-projects/gold-rhinoceros/ml-service/data/multilingual_dataset.jsonl")
    parser.add_argument("--translate",     action="store_true")
    parser.add_argument("--langs",         default="hi,ar,fr,es,zh,de,pt")
    parser.add_argument("--max-translate", type=int, default=500)
    args = parser.parse_args()

    all_rows = []

    print("Loading base:", args.base_jsonl)
    all_rows += load_base_jsonl(args.base_jsonl)
    all_rows += load_rubend18()
    all_rows += load_trustairlab()
    all_rows += load_jailbreakbench()

    print("\nTotal before dedup: {}".format(len(all_rows)))
    all_rows = deduplicate(all_rows)
    print("Total after dedup : {}".format(len(all_rows)))

    if args.translate:
        langs = [l.strip() for l in args.langs.split(",") if l.strip()]
        print("\nTranslating into:", langs)
        all_rows += translate_attacks(all_rows, langs, args.max_translate)
        all_rows  = deduplicate(all_rows)

    n_atk  = sum(1 for r in all_rows if r["label"] == 1)
    n_safe = sum(1 for r in all_rows if r["label"] == 0)
    lang_counts = {}
    for r in all_rows:
        lang_counts[r["lang"]] = lang_counts.get(r["lang"], 0) + 1

    print("\n" + "-" * 55)
    print("Final : {:,} rows | attack={:,} safe={:,}".format(
        len(all_rows), n_atk, n_safe))
    print("Langs :", lang_counts)
    if n_safe == 0:
        print("NOTE  : No safe rows — train.py auto-adds synthetic safe prompts.")
    print("-" * 55)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\nWritten  :", args.output)
    print("Retrain  :")
    print("  USE_GPU=1 python3 train.py \\")
    print("    --data-csv {} \\".format(args.output))
    print("    --architecture ensemble --threshold 0.35")


if __name__ == "__main__":
    main()
