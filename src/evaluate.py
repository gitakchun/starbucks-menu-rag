"""Retrieval·Generation 평가 (09장 방식을 고정 질문셋에 반복 적용)."""

import json
import re

from rag import NO_ANSWER, answer, format_docs, llm

# 14.4 평가용 테스트 질문 (스타벅스 코리아 메뉴·서비스 공개 자료)
# expected: 정답 근거에 반드시 들어 있어야 하는 키워드 목록 (모두 포함되어야 HIT)
# 문서에 없는 질문은 expected=None → Retrieval Hit Rate에서 제외합니다.
TEST_CASES = [
    {"id": "Q1", "type": "알레르기 직설 질문", "question": "우유 알레르기 있는데 카페 라떼 마셔도 돼요?", "expected": ["우유"]},
    {"id": "Q2", "type": "카페인 비교", "question": "카페 아메리카노랑 라떼 중에 카페인 적은 건 뭐예요?", "expected": ["150mg", "75mg"]},
    {"id": "Q3", "type": "표현이 다른 질문", "question": "우유 대신 식물성 음료 들어가는 커피 있어요?", "expected": ["오트음료", "오트 콜드 브루"]},
    {"id": "Q4", "type": "리워드 기본 질문", "question": "별 쌓으려면 어떻게 결제해야 해요?", "expected": ["등록된 스타벅스 카드", "간편결제"]},
    {"id": "Q5", "type": "영양 정보 질문", "question": "톨 라떼 칼로리가 얼마나 돼요?", "expected": ["180kcal"]},
    {"id": "Q6", "type": "복합 질문", "question": "톨 오트 콜드 브루 카페인이랑 당류가 얼마인지 알려주세요.", "expected": ["65mg", "14g"]},
    {"id": "Q7", "type": "공식 FAQ 질문", "question": "사이렌 오더에서 신용카드로 결제했는데 별 적립이 안 됐어요. 왜죠?", "expected": ["신용카드", "스타벅스 리워드 혜택"]},
    {"id": "Q8", "type": "알레르기 미표시 검증", "question": "아메리카노는 알레르기 성분이 하나도 없는 거 맞죠?", "expected": None},
    {"id": "Q9", "type": "알레르기 안전성 검증", "question": "오트 콜드 브루가 견과 알레르기 있어도 안전한지 보장해 주세요.", "expected": None},
    {"id": "Q10", "type": "문서에 없는 정보", "question": "이번 주말에 어느 매장에서 오트 콜드 브루 주문돼요?", "expected": None},
]


# ---------- Retrieval ----------

def hit(docs, expected: list[str]) -> int:
    """Top-K 전체에 기대 키워드가 모두 있으면 1입니다."""

    joined = "\n".join(doc.page_content for doc in docs)
    return 1 if all(keyword in joined for keyword in expected) else 0


def reciprocal_rank(docs, expected: list[str]) -> float:
    """상위부터 문서를 쌓아 기대 키워드가 모두 모이는 순위의 역수입니다."""

    seen = ""
    for rank, doc in enumerate(docs, start=1):
        seen += doc.page_content
        if all(keyword in seen for keyword in expected):
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(name: str, retriever) -> dict:
    """질문별 Top-K를 출력하고 Hit Rate·MRR을 계산합니다."""

    rows = []
    print(f"\n=== Retrieval: {name} ===")
    for case in TEST_CASES:
        docs = retriever.invoke(case["question"])
        row = {"case": case, "docs": docs, "hit": None, "rr": None}

        print(f"\n[{case['id']}] {case['question']}")
        if case["expected"] is None:
            print("판정: Hit Rate 평가 제외 (문서에 없는 질문)")
        else:
            row["hit"] = hit(docs, case["expected"])
            row["rr"] = reciprocal_rank(docs, case["expected"])
            print(f"판정: {'HIT' if row['hit'] else 'MISS'} / RR={row['rr']:.2f}")

        for i, doc in enumerate(docs, start=1):
            preview = doc.page_content.replace("\n", " ")[:70]
            score = doc.metadata.get("rerank_score")
            score_text = f" (rerank={score})" if score is not None else ""
            print(f"  [{i}] #{doc.metadata['chunk_id']}{score_text} {preview}")
        rows.append(row)

    scored = [row for row in rows if row["hit"] is not None]
    summary = {
        "hit_rate": sum(row["hit"] for row in scored) / len(scored),
        "mrr": sum(row["rr"] for row in scored) / len(scored),
    }
    print(f"\n{name} Hit Rate={summary['hit_rate']:.3f}, MRR={summary['mrr']:.3f}")
    return {"name": name, "rows": rows, **summary}


# ---------- Generation ----------

JUDGE_PROMPT = """
당신은 RAG 답변 평가자입니다. 문서, 질문, 답변을 보고 JSON 하나만 출력하세요.

- grounded: 답변 내용이 문서에 얼마나 근거하는지 1~5 정수
  (5=모두 문서에 근거, 3=일부 추측 섞임, 1=문서와 무관하거나 모순)
  "확인할 수 없다"고만 답한 경우 지어낸 내용이 없으므로 grounded=5로 평가합니다.
- answered: 답변이 질문에 직접 답했으면 true, 확인할 수 없다고 답했거나 일부만 답했으면 false

[문서]
{context}

[질문]
{question}

[답변]
{answer}

출력 예: {{"grounded": 5, "answered": true}}
"""


def judge(question: str, docs, answer_text: str) -> dict:
    response = llm.invoke(JUDGE_PROMPT.format(
        context=format_docs(docs),
        question=question,
        answer=answer_text,
    )).text
    match = re.search(r"\{.*\}", response, re.S)
    if not match:
        return {"grounded": 0, "answered": False}
    return json.loads(match.group())


def is_refusal(answer_text: str) -> bool:
    return "확인할 수 없" in answer_text


def evaluate_generation(result: dict) -> dict:
    """검색 결과 그대로 답변을 생성하고 근거성·직접 답변·거절 여부를 봅니다."""

    print(f"\n=== Generation: {result['name']} ===")
    for row in result["rows"]:
        case = row["case"]
        row["answer"] = answer(case["question"], row["docs"])
        row["judge"] = judge(case["question"], row["docs"], row["answer"])
        row["refused"] = is_refusal(row["answer"])
        # LLM Judge가 거절 답변을 "직접 답함"으로 채점하는 경우가 있어,
        # 확인할 수 없다고 답했으면 규칙으로 answered=False를 고정합니다.
        if row["refused"]:
            row["judge"]["answered"] = False
        print(f"\n[{case['id']}] {case['question']}")
        print("답변:", row["answer"])
        print("평가:", row["judge"], "| 거절:", row["refused"])

    answerable = [row for row in result["rows"] if row["case"]["expected"] is not None]
    unanswerable = [row for row in result["rows"] if row["case"]["expected"] is None]
    result["grounded_avg"] = sum(row["judge"]["grounded"] for row in answerable) / len(answerable)
    result["answered_rate"] = sum(bool(row["judge"]["answered"]) for row in answerable) / len(answerable)
    result["refusal_ok"] = all(row["refused"] for row in unanswerable)
    return result


# ---------- 결과 기록 ----------

def verdict(base_row: dict, improved_row: dict) -> str:
    """답변 변화를 먼저 보고, 답변이 같으면 검색 순위(RR) 변화로 판단합니다."""

    if base_row["case"]["expected"] is None:
        if base_row["refused"] == improved_row["refused"]:
            return "동일 (근거 없음 거절 유지)" if improved_row["refused"] else "동일"
        return "좋아짐 (답변)" if improved_row["refused"] else "나빠짐 (답변)"

    before = bool(base_row["judge"]["answered"])
    after = bool(improved_row["judge"]["answered"])
    if before != after:
        return "좋아짐 (답변)" if after else "나빠짐 (답변)"
    if improved_row["rr"] != base_row["rr"]:
        return "좋아짐 (검색 순위)" if improved_row["rr"] > base_row["rr"] else "나빠짐 (검색 순위)"
    return "동일"


def chunk_ids(docs) -> str:
    return ", ".join(f"#{doc.metadata['chunk_id']}" for doc in docs)


def short(text: str, limit: int = 60) -> str:
    text = text.replace("\n", " ").replace("|", "/")
    return text if len(text) <= limit else text[:limit] + "…"


def write_report(path: str, chunks, baseline: dict, improved: dict, conditions: dict, tried=()):
    """실행 결과를 results/run_result.md로 남깁니다 (사람이 쓰는 해석은 evaluation.md)."""

    lines = ["# 실행 결과 (자동 생성)", "", "`uv run python src/capstone_compare.py` 실행 시 덮어씁니다.", ""]
    lines += ["## 비교 조건", ""] + [f"- {key}: {value}" for key, value in conditions.items()] + [""]

    # 요약: Baseline, Retrieval만 측정한 시도안(tried), 최종 개선안 순서로 나란히 비교합니다.
    columns = [baseline, *tried, improved]

    def metric(result, key, fmt):
        if key not in result:
            return "-"
        value = result[key]
        if isinstance(value, bool):
            return "성공" if value else "실패"
        return format(value, fmt)

    scored_count = sum(row["hit"] is not None for row in baseline["rows"])
    lines += [
        "## 요약", "",
        "| 지표 | " + " | ".join(r["name"] for r in columns) + " |",
        "|---" * (len(columns) + 1) + "|",
    ]
    for label, key, fmt in [
        (f"Retrieval Hit Rate ({scored_count}문항)", "hit_rate", ".3f"),
        (f"Retrieval MRR ({scored_count}문항)", "mrr", ".3f"),
        ("Generation 근거성 평균 (1~5)", "grounded_avg", ".2f"),
        ("Generation 직접 답변 비율", "answered_rate", ".3f"),
        ("문서에 없는 질문 거절", "refusal_ok", ""),
    ]:
        lines.append(f"| {label} | " + " | ".join(metric(r, key, fmt) for r in columns) + " |")
    lines.append("")

    if tried:
        lines += [
            "## 시도안 질문별 RR", "",
            "| 질문 | Baseline | " + " | ".join(r["name"] for r in tried) + " |",
            "|---" * (len(tried) + 2) + "|",
        ]
        for i, base_row in enumerate(baseline["rows"]):
            if base_row["case"]["expected"] is None:
                continue
            cells = [f"{r['rows'][i]['rr']:.2f} ({chunk_ids(r['rows'][i]['docs'])})" for r in tried]
            lines.append(
                f"| {base_row['case']['id']} | {base_row['rr']:.2f} ({chunk_ids(base_row['docs'])}) | "
                + " | ".join(cells) + " |"
            )
        lines.append("")

    lines += [
        "## 질문별 결과", "",
        "| 질문 | 유형 | Baseline Top-K | 개선 Top-K | 정답 포함 (전→후) | RR (전→후) | 답변 변화 | 판단 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for base_row, imp_row in zip(baseline["rows"], improved["rows"]):
        case = base_row["case"]
        if case["expected"] is None:
            included, rr = "평가 제외", "-"
        else:
            included = f"{'O' if base_row['hit'] else 'X'} → {'O' if imp_row['hit'] else 'X'}"
            rr = f"{base_row['rr']:.2f} → {imp_row['rr']:.2f}"
        change = "동일" if base_row["answer"] == imp_row["answer"] else "변경"
        lines.append(
            f"| {case['id']} {case['question']} | {case['type']} | {chunk_ids(base_row['docs'])} | "
            f"{chunk_ids(imp_row['docs'])} | {included} | {rr} | {change} | {verdict(base_row, imp_row)} |"
        )

    lines += ["", "## 질문별 답변", ""]
    for base_row, imp_row in zip(baseline["rows"], improved["rows"]):
        lines += [
            f"### {base_row['case']['id']}. {base_row['case']['question']}", "",
            f"- Baseline ({base_row['judge']}): {short(base_row['answer'], 200)}",
            f"- 개선 ({imp_row['judge']}): {short(imp_row['answer'], 200)}",
            "",
        ]

    lines += ["## Chunk 목록", "", "| ID | 내용 |", "|---|---|"]
    lines += [f"| #{c.metadata['chunk_id']} | {short(c.page_content, 90)} |" for c in chunks]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n결과를 {path}에 기록했습니다.")
