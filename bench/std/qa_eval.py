#!/usr/bin/env python3
"""End-to-end QA on LongMemEval_S and LoCoMo from unimem-std-bench retrieval output (NOT run: needs an API key).

Pipeline (both datasets): retrieval  ->  reader LLM (any OpenAI-compatible /v1/chat/completions)  ->  scoring.

1. Retrieval for every question (incl. LongMemEval abstention and LoCoMo adversarial questions):

       unimem-std-bench dump --cache $C --dataset lme    --gran turn --system unimem-prior-e5 --k 50 --out lme.jsonl
       unimem-std-bench dump --cache $C --dataset locomo --gran turn --system unimem-prior-e5 --k 25 --out locomo.jsonl

   (`--system` = any system of the retrieval bench: bm25, e5-small, hybrid-rrf, unimem-e5, unimem-prior-e5, …)

2. QA + scoring:

       export OPENAI_API_KEY=...   [OPENAI_BASE_URL=https://.../v1]
       python3 bench/std/qa_eval.py lme    --cache $C --retrieval lme.jsonl    --topk 10 --reader gpt-4o-2024-08-06 --judge gpt-4o-2024-08-06
       python3 bench/std/qa_eval.py locomo --cache $C --retrieval locomo.jsonl --topk 10 --reader gpt-4o-mini

   `--dry-run` builds every prompt and prints one, without any network call (used to test this script offline).
   `--limit N` evaluates only the first N questions. Outputs `<retrieval>.qa-<reader>.jsonl` (resumable) and a
   summary JSON next to it.

LongMemEval reading follows the official `src/generation/run_generation.py` for `flat-turn` / `flat-session`
retrieval with `history_format=json`, `useronly=false`, no chain-of-note: a retrieved user turn is expanded to its
round (user turn + the following assistant turn), the top-k chunks are sorted by session date and rendered as
"### Session i: / Session Date: / Session Content:" blocks, and the answer prompt is the official one (with
`--cot`, the step-by-step variant). Scoring reproduces the official GPT-4o judge prompts of
`src/evaluation/evaluate_qa.py` (`get_anscheck_prompt`) verbatim, per question type and for abstention (`_abs`),
temperature 0, max_tokens 10, label = 'yes' in the reply. The paper's judge is `gpt-4o-2024-08-06`.

LoCoMo reading follows `task_eval/gpt_utils.py` + `rag_utils.py` (RAG over dialogs): context lines
"<session date>: <speaker> said, "<text>"" (+ "[shares <caption>]") of the top-k retrieved turns in rank order, then
QA_PROMPT (category 5: QA_PROMPT_CAT_5 with the official two-option "(a)/(b)" adversarial format and a seeded coin
flip); category 2 questions get " Use DATE of CONVERSATION to answer with an approximate date." Scoring is the
official `task_eval/evaluation.py`: token F1 with Porter stemming (multi-hop: per-sub-answer partial F1; category 3:
first ';'-separated gold answer; category 5: 1 if the prediction says "no information available" / "not mentioned").
The Porter stemmer is NLTK's when installed, otherwise a faithful fallback of the original algorithm.
"""
import argparse
import json
import os
import random
import re
import string
import sys
import time
import urllib.request
from collections import Counter, defaultdict

# --------------------------------------------------------------------------- official LongMemEval prompts

LME_ANSWER_PROMPT = (
    "I will give you several history chats between you and a user. Please answer the question based on the relevant chat history."
    "\n\n\nHistory Chats:\n\n{}\n\nCurrent Date: {}\nQuestion: {}\nAnswer:"
)
LME_ANSWER_PROMPT_COT = (
    "I will give you several history chats between you and a user. Please answer the question based on the relevant chat history. "
    "Answer the question step by step: first extract all the relevant information, and then reason over the information to get the answer."
    "\n\n\nHistory Chats:\n\n{}\n\nCurrent Date: {}\nQuestion: {}\nAnswer (step by step):"
)


def get_anscheck_prompt(task, question, answer, response, abstention=False):
    """Verbatim from LongMemEval src/evaluation/evaluate_qa.py."""
    if not abstention:
        if task in ["single-session-user", "single-session-assistant", "multi-session"]:
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == "temporal-reasoning":
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == "knowledge-update":
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == "single-session-preference":
            template = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        else:
            raise NotImplementedError
    else:
        template = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
        prompt = template.format(question, answer, response)
    return prompt


# --------------------------------------------------------------------------- official LoCoMo prompts

LOCOMO_CONV_START = "Below is a conversation between two people: {} and {}. The conversation takes place over multiple days and the date of each conversation is wriiten at the beginning of the conversation.\n\n"
LOCOMO_QA_PROMPT = """
Based on the above context, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {} Short answer:
"""
LOCOMO_QA_PROMPT_CAT_5 = """
Based on the above context, answer the following question.

Question: {} Short answer:
"""

# --------------------------------------------------------------------------- LoCoMo scoring (task_eval/evaluation.py)

try:
    from nltk.stem import PorterStemmer

    _ps = PorterStemmer()
    stem = _ps.stem
except Exception:  # pragma: no cover - fallback when nltk is not installed
    def stem(w):
        """Porter (1980) stemmer, compact implementation (NLTK's default mode adds a few extensions)."""
        w = w.lower()
        if len(w) <= 2:
            return w
        V = "aeiou"

        def cons(s, i):
            c = s[i]
            if c in V:
                return False
            if c == "y":
                return i == 0 or not cons(s, i - 1)
            return True

        def m(s):
            n, i, L = 0, 0, len(s)
            while i < L and cons(s, i):
                i += 1
            while i < L:
                while i < L and not cons(s, i):
                    i += 1
                if i >= L:
                    break
                n += 1
                while i < L and cons(s, i):
                    i += 1
            return n

        def hasv(s):
            return any(not cons(s, i) for i in range(len(s)))

        def dbl(s):
            return len(s) >= 2 and s[-1] == s[-2] and cons(s, len(s) - 1)

        def cvc(s):
            return len(s) >= 3 and cons(s, len(s) - 3) and not cons(s, len(s) - 2) and cons(s, len(s) - 1) and s[-1] not in "wxy"

        if w.endswith("sses"):
            w = w[:-2]
        elif w.endswith("ies"):
            w = w[:-2]
        elif w.endswith("ss"):
            pass
        elif w.endswith("s"):
            w = w[:-1]
        flag = False
        if w.endswith("eed"):
            if m(w[:-3]) > 0:
                w = w[:-1]
        elif w.endswith("ed") and hasv(w[:-2]):
            w, flag = w[:-2], True
        elif w.endswith("ing") and hasv(w[:-3]):
            w, flag = w[:-3], True
        if flag:
            if w.endswith(("at", "bl", "iz")):
                w += "e"
            elif dbl(w) and w[-1] not in "lsz":
                w = w[:-1]
            elif m(w) == 1 and cvc(w):
                w += "e"
        if w.endswith("y") and hasv(w[:-1]):
            w = w[:-1] + "i"
        for suf, rep in [("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"), ("izer", "ize"), ("abli", "able"), ("alli", "al"),
                         ("entli", "ent"), ("eli", "e"), ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"), ("ator", "ate"), ("alism", "al"),
                         ("iveness", "ive"), ("fulness", "ful"), ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble")]:
            if w.endswith(suf):
                if m(w[: -len(suf)]) > 0:
                    w = w[: -len(suf)] + rep
                break
        for suf, rep in [("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"), ("ical", "ic"), ("ful", ""), ("ness", "")]:
            if w.endswith(suf):
                if m(w[: -len(suf)]) > 0:
                    w = w[: -len(suf)] + rep
                break
        for suf in ["al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment", "ent", "ion", "ou", "ism", "ate", "iti", "ous", "ive", "ize"]:
            if w.endswith(suf):
                st = w[: -len(suf)]
                if m(st) > 1 and (suf != "ion" or st.endswith(("s", "t"))):
                    w = st
                break
        if w.endswith("e"):
            st = w[:-1]
            if m(st) > 1 or (m(st) == 1 and not cvc(st)):
                w = st
        if m(w) > 1 and dbl(w) and w.endswith("l"):
            w = w[:-1]
        return w


def normalize_answer(s):
    s = s.replace(",", "")
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the|and)\b", " ", s)
    return " ".join(s.split())


def f1_score(prediction, ground_truth):
    p = [stem(w) for w in normalize_answer(prediction).split()]
    g = [stem(w) for w in normalize_answer(ground_truth).split()]
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def f1_multi(prediction, ground_truth):
    preds = [x.strip() for x in prediction.split(",")]
    gts = [x.strip() for x in ground_truth.split(",")]
    return sum(max(f1_score(p, g) for p in preds) for g in gts) / len(gts)


def locomo_score(cat, prediction, answer):
    answer = str(answer)
    if cat == 3:
        answer = answer.split(";")[0].strip()
    if cat in (2, 3, 4):
        return f1_score(prediction, answer)
    if cat == 1:
        return f1_multi(prediction, answer)
    if cat == 5:
        o = prediction.lower()
        return 1.0 if ("no information available" in o or "not mentioned" in o) else 0.0
    raise ValueError(cat)


def get_cat_5_answer(model_prediction, answer_key):
    p = model_prediction.strip().lower()
    if len(p) == 1:
        return answer_key["a"] if "a" in p else answer_key["b"]
    if len(p) == 3:
        return answer_key["a"] if "(a)" in p else answer_key["b"]
    return p


# --------------------------------------------------------------------------- OpenAI-compatible client


def chat(model, prompt, max_tokens, temperature=0.0, retries=6):
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        sys.exit("OPENAI_API_KEY is not set (use --dry-run to only build prompts)")
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], "n": 1, "temperature": temperature, "max_tokens": max_tokens}).encode()
    for i in range(retries):
        try:
            req = urllib.request.Request(base + "/chat/completions", data=body, headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)["choices"][0]["message"]["content"].strip()
        except Exception as e:  # rate limits / transient errors: exponential backoff
            if i == retries - 1:
                raise
            time.sleep(2 ** i)
            print(f"retry {i + 1}: {e}", file=sys.stderr)


# --------------------------------------------------------------------------- LongMemEval


def lme_prompt(entry, ranked, topk, cot):
    sid2date = dict(zip(entry["haystack_session_ids"], entry["haystack_dates"]))
    sid2sess = dict(zip(entry["haystack_session_ids"], entry["haystack_sessions"]))
    chunks = []
    for cid in ranked:
        if cid in sid2sess:  # session-granularity id
            chunks.append((sid2date[cid], [{k: v for k, v in t.items() if k != "has_answer"} for t in sid2sess[cid]]))
        else:  # "<session>_<turn#>" (1-based): expand the turn into its round
            sid, t = cid.rsplit("_", 1)
            t = int(t) - 1
            sess = sid2sess.get(sid)
            if sess is None or t >= len(sess):
                continue
            rnd = [sess[t]] + ([sess[t + 1]] if t + 1 < len(sess) else [])
            chunks.append((sid2date[sid], [{k: v for k, v in x.items() if k != "has_answer"} for x in rnd]))
        if len(chunks) >= topk:
            break
    chunks.sort(key=lambda x: x[0])
    hist = ""
    for i, (date, ent) in enumerate(chunks):
        hist += "\n### Session {}:\nSession Date: {}\nSession Content:\n{}\n".format(i + 1, date, "\n" + json.dumps(ent))
    tmpl = LME_ANSWER_PROMPT_COT if cot else LME_ANSWER_PROMPT
    return tmpl.format(hist, entry["question_date"], entry["question"])


def run_lme(a):
    data = {e["question_id"]: e for e in json.load(open(os.path.join(a.cache, "data/longmemeval_s_cleaned.json")))}
    ret = [json.loads(l) for l in open(a.retrieval)]
    if a.limit:
        ret = ret[: a.limit]
    out_path = f"{a.retrieval}.qa-{a.reader.replace('/', '_')}.jsonl"
    done = {}
    if os.path.exists(out_path) and not a.dry_run:
        done = {r["question_id"]: r for r in map(json.loads, open(out_path))}
    out = None if a.dry_run else open(out_path, "a")
    for r in ret:
        e = data[r["qid"]]
        prompt = lme_prompt(e, r["ranked"], a.topk, a.cot)
        if a.dry_run:
            continue
        if r["qid"] in done:
            continue
        hyp = chat(a.reader, prompt, max_tokens=a.max_tokens)
        judge_prompt = get_anscheck_prompt(e["question_type"], e["question"], e["answer"], hyp, abstention="_abs" in e["question_id"])
        verdict = chat(a.judge, judge_prompt, max_tokens=10)
        rec = {"question_id": r["qid"], "question_type": e["question_type"], "hypothesis": hyp, "autoeval_label": {"model": a.judge, "label": "yes" in verdict.lower()}}
        out.write(json.dumps(rec) + "\n")
        out.flush()
        done[r["qid"]] = rec
    if a.dry_run:
        print(lme_prompt(data[ret[0]["qid"]], ret[0]["ranked"], a.topk, a.cot)[:3000])
        print(f"\n[dry-run] built {len(ret)} LongMemEval prompts")
        return
    by = defaultdict(list)
    for q, rec in done.items():
        t = "abstention" if "_abs" in q else rec["question_type"]
        by[t].append(1.0 if rec["autoeval_label"]["label"] else 0.0)
    allv = [v for vs in by.values() for v in vs]
    summ = {"reader": a.reader, "judge": a.judge, "topk": a.topk, "cot": a.cot, "retrieval": a.retrieval, "n": len(allv),
            "accuracy": sum(allv) / len(allv), "by_type": {t: {"n": len(v), "acc": sum(v) / len(v)} for t, v in sorted(by.items())}}
    json.dump(summ, open(out_path.replace(".jsonl", ".summary.json"), "w"), indent=2)
    print(json.dumps(summ, indent=2))


# --------------------------------------------------------------------------- LoCoMo


def run_locomo(a):
    data = json.load(open(os.path.join(a.cache, "data/locomo10.json")))
    conv = {c["sample_id"]: c for c in data}
    ret = [json.loads(l) for l in open(a.retrieval)]
    if a.limit:
        ret = ret[: a.limit]
    rng = random.Random(0)
    out_path = f"{a.retrieval}.qa-{a.reader.replace('/', '_')}.jsonl"
    done = {}
    if os.path.exists(out_path) and not a.dry_run:
        done = {r["qid"]: r for r in map(json.loads, open(out_path))}
    out = None if a.dry_run else open(out_path, "a")
    first = None
    for r in ret:
        sample, i = r["qid"].split("#")
        c = conv[sample]
        qa = c["qa"][int(i)]
        cv = c["conversation"]
        turns = {}
        for k, v in cv.items():
            if k.startswith("session_") and isinstance(v, list):
                date = cv[k + "_date_time"]
                for t in v:
                    s = t["speaker"] + ' said, "' + t["text"] + '"' + "\n"
                    if t.get("blip_caption"):
                        s += "[shares %s]\n" % t["blip_caption"]
                    turns[t["dia_id"]] = (date, s)
        ctx = "\n".join(d + ": " + s for d, s in (turns[x] for x in r["ranked"][: a.topk] if x in turns))
        cat = qa["category"]
        answer_key = None
        if cat == 2:
            q = qa["question"] + " Use DATE of CONVERSATION to answer with an approximate date."
        elif cat == 5:
            q = qa["question"] + " Select the correct answer: (a) {} (b) {}. "
            adv = qa.get("adversarial_answer", qa.get("answer", ""))
            if rng.random() < 0.5:
                q = q.format("Not mentioned in the conversation", adv)
                answer_key = {"a": "Not mentioned in the conversation", "b": adv}
            else:
                q = q.format(adv, "Not mentioned in the conversation")
                answer_key = {"b": "Not mentioned in the conversation", "a": adv}
        else:
            q = qa["question"]
        prompt = LOCOMO_CONV_START.format(cv["speaker_a"], cv["speaker_b"]) + ctx + "\n\n" + (LOCOMO_QA_PROMPT_CAT_5 if cat == 5 else LOCOMO_QA_PROMPT).format(q)
        first = first or prompt
        if a.dry_run or r["qid"] in done:
            continue
        pred = chat(a.reader, prompt, max_tokens=a.max_tokens)
        if cat == 5:
            pred = get_cat_5_answer(pred, answer_key)
        gold = qa.get("answer", "")
        rec = {"qid": r["qid"], "category": cat, "prediction": pred, "answer": gold, "score": locomo_score(cat, pred, gold),
               "evidence": qa.get("evidence", []), "context_ids": r["ranked"][: a.topk]}
        out.write(json.dumps(rec) + "\n")
        out.flush()
        done[r["qid"]] = rec
    if a.dry_run:
        print(first[:3000])
        print(f"\n[dry-run] built {len(ret)} LoCoMo prompts")
        return
    by = defaultdict(list)
    for rec in done.values():
        by[rec["category"]].append(rec["score"])
    names = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
    allv = [v for vs in by.values() for v in vs]
    summ = {"reader": a.reader, "topk": a.topk, "retrieval": a.retrieval, "n": len(allv), "f1_overall": sum(allv) / len(allv),
            "by_category": {names[c]: {"n": len(v), "f1": sum(v) / len(v)} for c, v in sorted(by.items())}}
    json.dump(summ, open(out_path.replace(".jsonl", ".summary.json"), "w"), indent=2)
    print(json.dumps(summ, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=["lme", "locomo"])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--retrieval", required=True, help="output of `unimem-std-bench dump`")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--reader", default="gpt-4o-2024-08-06")
    ap.add_argument("--judge", default="gpt-4o-2024-08-06", help="LongMemEval judge (the paper: gpt-4o-2024-08-06)")
    ap.add_argument("--cot", action="store_true", help="LongMemEval step-by-step answer prompt")
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.max_tokens is None:
        a.max_tokens = 500 if a.dataset == "lme" else 50
    (run_lme if a.dataset == "lme" else run_locomo)(a)


if __name__ == "__main__":
    main()
