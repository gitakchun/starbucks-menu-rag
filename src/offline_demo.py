"""Run retrieval and inspect evidence without external model APIs."""

import argparse
from pathlib import Path
import re

from langchain_community.retrievers import BM25Retriever

from ingest import load_documents, split_documents

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "starbucks_knowledge.jsonl"
TOP_K = 3


def tokenize_ko(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*|[가-힣]+", text.lower())
    prefixes = [
        token[:length]
        for token in tokens
        if re.fullmatch(r"[가-힣]+", token)
        for length in range(2, min(len(token), 4) + 1)
    ]
    return tokens + prefixes


def main():
    parser = argparse.ArgumentParser(
        description="API 키 없이 메뉴·FAQ 근거를 로컬 검색합니다. 생성형 답변은 만들지 않습니다."
    )
    parser.add_argument(
        "--question",
        action="append",
        help="검색 질문. 여러 번 지정할 수 있으며 생략하면 샘플 질문을 실행합니다.",
    )
    args = parser.parse_args()

    docs = load_documents(str(DATA_PATH))
    chunks = split_documents(docs)
    retriever = BM25Retriever.from_documents(
        chunks,
        preprocess_func=tokenize_ko,
    )
    retriever.k = TOP_K

    questions = args.question or [
        "우유 알레르기가 있는데 카페 라떼에 우유가 들어가나요?",
        "카페인 적은 음료가 있어요?",
        "사이렌 오더에서 신용카드로 결제했는데 별 적립이 안 됐어요. 왜죠?",
        "아메리카노는 알레르기 성분이 하나도 없나요?",
    ]

    print(f"문서 {len(docs)}개 → Chunk {len(chunks)}개 | BM25 Top-{TOP_K}")
    print("오프라인 모드: 아래는 검색 근거이며, LLM이 만든 답변이 아닙니다.")
    for question in questions:
        print(f"\n질문: {question}")
        results = retriever.invoke(question)
        if not results:
            print("검색 결과가 없습니다.")
            continue
        for rank, doc in enumerate(results, start=1):
            print(f"\n[{rank}] {doc.page_content[:900]}")
            print(f"출처: {doc.metadata.get('source', '출처 없음')}")
            if doc.metadata.get("allergen_status") == "not_shown":
                print("주의: 알레르기 정보 미표시는 무알레르기 또는 안전을 뜻하지 않습니다.")


if __name__ == "__main__":
    main()