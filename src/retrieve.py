"""RUNTIME 검색: Baseline(Dense), 시도안(Hybrid), 최종 개선안(Dense + LLM Reranker)."""

import re

from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.runnables import RunnableLambda

from rag import llm

TOP_K = 3
RERANK_FETCH_K = 6
DENSE_WEIGHT = 0.5
SPARSE_WEIGHT = 0.5


def build_baseline_retriever(vectorstore, k: int = TOP_K):
    """Embedding 유사도만 사용하는 Baseline Retriever입니다."""

    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


# ---------- 시도안: BM25 + Dense Hybrid ----------

def tokenize_ko(text: str) -> list[str]:
    """BM25용 토크나이저입니다.

    기본 BM25는 공백으로만 나누므로 "E-410은"처럼 조사가 붙은 코드를
    찾지 못합니다. 영문·숫자 코드는 하이픈째로 하나의 토큰으로 떼어 내고,
    한글은 어절 그대로 둡니다.
    """

    return re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*|[가-힣]+", text.upper())


def build_hybrid_retriever(vectorstore, chunks, k: int = TOP_K):
    """Dense(의미)와 Sparse(키워드) 결과를 RRF로 합친 뒤 Top-K만 남깁니다."""

    dense = vectorstore.as_retriever(search_kwargs={"k": k})

    sparse = BM25Retriever.from_documents(chunks, preprocess_func=tokenize_ko)
    sparse.k = k

    ensemble = EnsembleRetriever(
        retrievers=[dense, sparse],
        weights=[DENSE_WEIGHT, SPARSE_WEIGHT],
    )

    # EnsembleRetriever는 두 결과의 합집합(최대 2k개)을 돌려주므로,
    # Baseline과 같은 조건으로 비교하려고 상위 k개로 자릅니다.
    return ensemble | RunnableLambda(lambda docs: docs[:k])


# ---------- 최종 개선안: Dense 후보 → LLM Reranker ----------

RELEVANCE_PROMPT = """
다음 질문에 답하는 데 아래 문서가 얼마나 직접적으로 도움이 되는지 0~100 사이 정수 하나로 평가하세요.
질문의 답이 문서에 직접 들어 있으면 높은 점수, 주제만 비슷하거나 제목뿐이면 낮은 점수를 주세요.
설명하지 말고 숫자만 출력하세요.

[질문]
{question}

[문서]
{document}
"""


def relevance_scores(question: str, docs) -> list[int]:
    """후보 문서마다 질문과의 관련도를 0~100으로 매깁니다 (병렬 호출)."""

    responses = llm.batch([
        RELEVANCE_PROMPT.format(question=question, document=doc.page_content)
        for doc in docs
    ])
    scores = []
    for response in responses:
        match = re.search(r"\d+", response.text)
        scores.append(min(100, max(0, int(match.group()))) if match else 0)
    return scores


def build_rerank_retriever(vectorstore, k: int = TOP_K, fetch_k: int = RERANK_FETCH_K):
    """Dense로 후보 fetch_k개를 넓게 가져온 뒤 LLM 관련도 순으로 상위 k개만 남깁니다."""

    candidates = vectorstore.as_retriever(search_kwargs={"k": fetch_k})

    def rerank(question: str):
        docs = candidates.invoke(question)
        scores = relevance_scores(question, docs)
        # 점수가 같으면 원래 Dense 순위를 유지합니다 (sorted는 안정 정렬).
        ranked = sorted(zip(scores, docs), key=lambda pair: pair[0], reverse=True)
        # Vector Store에 저장된 원본 Document를 건드리지 않도록 복사본에 점수를 기록합니다.
        return [
            doc.model_copy(update={"metadata": {**doc.metadata, "rerank_score": score}})
            for score, doc in ranked[:k]
        ]

    return RunnableLambda(rerank)
