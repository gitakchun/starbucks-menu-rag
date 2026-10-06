from dotenv import load_dotenv

from evaluate import TEST_CASES, evaluate_generation, evaluate_retrieval, write_report
from ingest import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    build_vectorstore,
    load_documents,
    split_documents,
)
from rag import CHAT_MODEL, answer_with_retriever
from retrieve import (
    DENSE_WEIGHT,
    RERANK_FETCH_K,
    SPARSE_WEIGHT,
    TOP_K,
    build_baseline_retriever,
    build_hybrid_retriever,
    build_rerank_retriever,
)

load_dotenv()

# TODO 1 (14.2) 문제 정의와 문서 범위
# 스타벅스 코리아 공식 공개 자료를 근거로 메뉴·영양·알레르기·서비스 질문에 답한다.
# 알레르기 정보가 빠져 있거나 문서에 근거가 없으면 안전을 단정하지 않고 답변을 보류한다.
DATA_PATH = "data/starbucks_knowledge.jsonl"

# TODO 2 (14.3) Baseline RAG: Loader → Splitter → Embedding → Vector Store → Retriever
docs = load_documents(DATA_PATH)
chunks = split_documents(docs)
vectorstore = build_vectorstore(chunks)
baseline = build_baseline_retriever(vectorstore)

print(f"문서 {len(docs)}개 → Chunk {len(chunks)}개")
for chunk in chunks:
    preview = chunk.page_content.replace("\n", " ")[:60]
    print(f"  #{chunk.metadata['chunk_id']} ({len(chunk.page_content)}자) {preview}")

# TODO 3 (14.4) 고정 질문셋: 메뉴·카페인·알레르기·리워드의 답변 가능/불가 질문
print(f"\n테스트 질문 {len(TEST_CASES)}개")

# TODO 4 (14.5, 14.8) Baseline 결과를 먼저 기록
baseline_result = evaluate_retrieval("Baseline Similarity", baseline)

# TODO 5 (14.6) 개선 전략
# Dense 검색과 BM25 Hybrid, Dense 후보 확장 후 LLM Reranker를 같은 질문셋으로 비교합니다.
# 특히 알레르기 정보가 없는 질문에서 검색 문맥이 안전성 보증으로 오해되지 않는지 확인합니다.
hybrid = build_hybrid_retriever(vectorstore, chunks)
improved = build_rerank_retriever(vectorstore)

# TODO 6 (14.7, 14.8) 같은 질문셋으로 Retrieval·Generation 비교
hybrid_result = evaluate_retrieval("시도: Hybrid", hybrid)
improved_result = evaluate_retrieval("개선: Reranker", improved)
evaluate_generation(baseline_result)
evaluate_generation(improved_result)

print("\n=== 비교 ===")
for key in ["hit_rate", "mrr", "grounded_avg", "answered_rate", "refusal_ok"]:
    hybrid_value = hybrid_result.get(key, "-")
    print(f"{key}: Baseline={baseline_result[key]} / Hybrid={hybrid_value} / Reranker={improved_result[key]}")

print("\n=== 개선 Pipeline 최종 답변 예시 ===")
for question in ["카페인 적은 음료가 뭐예요?", "카페 아메리카노 알레르기 성분 없죠?"]:
    print("질문:", question)
    print("답변:", answer_with_retriever(question, improved))

# TODO 7 (14.9~14.13) 실행 결과를 파일로 남기고 README·design·evaluation에 해석을 정리
write_report(
    "results/run_result.md",
    chunks,
    baseline_result,
    improved_result,
    conditions={
        "문서": f"`{DATA_PATH}`",
        "Chunk": f"chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP} → {len(chunks)}개",
        "Embedding": EMBEDDING_MODEL,
        "Vector Store": "FAISS",
        "Top-K": TOP_K,
        "Baseline": "Dense Similarity Search",
        "시도": f"EnsembleRetriever(Dense {DENSE_WEIGHT} + BM25 {SPARSE_WEIGHT}, RRF) → 상위 {TOP_K}개",
        "개선": f"Dense 후보 {RERANK_FETCH_K}개 → LLM 관련도(0~100) Rerank → 상위 {TOP_K}개",
        "Chat Model": f"{CHAT_MODEL} (temperature=0, 답변·Rerank·Judge 공통)",
    },
    tried=[hybrid_result],
)
