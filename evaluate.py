"""Оценка решения: python evaluate.py <answers.jsonl> <labels.jsonl>

answers.jsonl — по строке на документ:
  {"doc_id": "...", "decision": "accept" | "reject", "fields": {<10 полей>}}
Поля нужны только у принятых документов.

Принятый документ засчитывается, если все 10 полей совпадают с разметкой (сравнение без
учёта регистра и повторных пробелов).

Действительность паспорта этот скрипт НЕ проверяет: её учитываем при итоговой оценке на test.
"""
import json
import sys

FIELDS = ["Last_name_ru", "First_name_ru", "Middle_name_ru", "Sex_ru", "Birth_date",
          "Birth_place_ru", "Issue_date", "Issue_organisation_code", "Licence_number",
          "Issue_organization_ru"]


def norm(s):
    return " ".join(str(s or "").split()).upper()


def load(path):
    with open(path, encoding="utf-8") as fh:
        return {d["doc_id"]: d for d in map(json.loads, fh)}


def main(ans_path, gt_path):
    ans, gt = load(ans_path), load(gt_path)
    missing = set(gt) - set(ans)
    if missing:
        sys.exit(f"нет ответа для {len(missing)} документов, например {sorted(missing)[:3]}")
    good = bad = 0
    field_errors = {f: 0 for f in FIELDS}
    for doc_id, g in gt.items():
        gf = g["fields"]
        a = ans[doc_id]
        if a.get("decision") != "accept":
            continue
        af = a.get("fields") or {}
        wrong = [f for f in FIELDS if norm(af.get(f)) != norm(gf[f])]
        for f in wrong:
            field_errors[f] += 1
        if wrong:
            bad += 1
        else:
            good += 1
    n = len(gt)
    print(f"документов:              {n}")
    print(f"принято верно:           {good}  ({good / n:.1%})")
    print(f"принято с ошибкой:       {bad}")
    print(f"ошибки по полям:         " + ", ".join(f"{f}={c}" for f, c in field_errors.items() if c))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
