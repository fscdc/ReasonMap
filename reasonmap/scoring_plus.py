"""Scorer for ReasonMap-Plus (counting and true/false questions)."""

from .math_answer import find_math_answer

_ANSWER_FLAGS = ("the final answer is", "the answer is", "the correct answer is", "the answer should be")
_CHOICES = {"a": 0, "b": 1, "c": 2, "d": 3}


def extract_plus_answer(final_answer):
    """Pull the short answer (number / letter / yes-no) out of a Plus generation."""
    model_answer = final_answer or ""
    if "oxed{" not in model_answer:
        for flag in _ANSWER_FLAGS:
            for variant in (flag, flag.replace("the", "The")):
                raw = model_answer
                model_answer = model_answer.split(variant)[-1].strip()
                if variant in raw:
                    model_answer = model_answer.split("\n")[0].split(". ")[0]
    elif model_answer.count("oxed{") > 1:
        model_answer = "\\boxed{" + model_answer.split("oxed{")[-1]
    return find_math_answer(model_answer).rstrip(".").lstrip(":").strip()


def is_correct(extracted, ground_truth, q_type):
    ground_truth = int(ground_truth)
    if "TorF" in q_type:
        return extracted.lower() == ("yes" if ground_truth == 1 else "no")
    if q_type in ("Counting2", "Counting3"):
        try:
            return int(extracted) == ground_truth
        except (ValueError, TypeError):
            return False
    if q_type == "Counting1":
        letter = extracted.lower()
        return letter in _CHOICES and _CHOICES[letter] == ground_truth
    return False


def score_plus_record(record):
    """Score one Plus prediction record and return the per-sample result JSON."""
    meta = record["meta"]
    gen = record["generations"]["question"]
    extracted = extract_plus_answer(gen["final_answer"])
    correct = is_correct(extracted, meta["answer"], meta["type"])
    return {
        "model": record["model"],
        "country": meta["country"],
        "city": meta["city"],
        "type": meta["type"],
        "question": meta["question"],
        "full answer": gen["response"],
        "token count": gen["token_count"],
        "final answer": gen["final_answer"],
        "extracted answer": extracted,
        "ground truth": meta["answer"],
        # key kept as "acc short" for compatibility with existing result files / cal_metrics
        "acc short": int(correct),
        "difficulty city": meta.get("difficulty_city"),
        "city line count": meta.get("city_line_count"),
        "city transfer count": meta.get("city_transfer_count"),
    }
