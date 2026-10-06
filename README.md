# 스타벅스 메뉴·알레르기 Q&A RAG

스타벅스 코리아 공식 공개 자료를 근거로 메뉴 영양·카페인·알레르기 표시와 리워드·사이렌 오더의 기본 안내를 제공하는 고객용 RAG입니다.
기존 파이프라인의 Dense / Hybrid / LLM Reranker 비교 구조를 유지하고, 알레르기 정보가 불충분할 때 안전을 단정하지 않는지 평가합니다.

## 1. 문제 정의

- 사용자는 "우유 못 먹는데 뭐 마셔요?", "카페인 적은 거 있어요?", "별 적립 왜 안 돼요?"처럼 일상적인 표현으로 묻습니다.
- 답변은 문서에 적힌 사실과 메뉴 규격에 한정합니다. 알레르기 표시가 없다고 해서 알레르기 성분이 없거나 안전하다고 추정하지 않습니다.
- 교차접촉과 개인별 안전성은 공식 메뉴 표만으로 보장할 수 없으므로 매장 파트너 확인을 안내합니다.

자료 출처·범위와 안전 기준은 [results/starbucks_design.md](results/starbucks_design.md), 평가 질문과 검증 기준은 [results/starbucks_evaluation.md](results/starbucks_evaluation.md)에 있습니다.

## 2. 사용 데이터

| 파일 | 내용 |
|---|---|
| `data/starbucks_knowledge.jsonl` | 공식 메뉴 상세 및 FAQ에서 수집한 구조화 레코드. 음료·푸드·FAQ 원문 출처와 수집일 포함 |

공식 목록에서 관찰한 상세 링크는 음료 198개, 푸드 118개입니다. 수집기는 카테고리 JSON과 FAQ 페이지 API를 읽고 코드/ID로 중복을 제거합니다. 현재 판매 가능성은 목록 노출 범위만 뜻하며, 매장별 실시간 재고를 보장하지 않습니다.

## 3. RAG 구조

```text
[PREPROCESSING]
starbucks_knowledge.jsonl → JSONL Documents → RecursiveCharacterTextSplitter(450/40) → OpenAIEmbeddings → FAISS

[RUNTIME]
Question → Dense Retriever (후보 6개) → LLM Reranker (관련도 0~100) → Top-3
                                                                       ↓
                                                    Context → Prompt → gpt-4o-mini → Answer
```

- Baseline: Dense Similarity Retriever (k=3)
- 시도했지만 채택하지 않은 방식: Dense + BM25 Hybrid
- 최종 개선: 위 구조의 Dense + Reranker

```text
starbucks-menu-rag/
├── .env.example
├── .gitignore
├── .python-version           # Python 3.11
├── pyproject.toml
├── uv.lock                   # 의존성 버전 고정
├── README.md
├── data/
│   ├── menu_and_service_faq.txt  # 초기 3종 샘플
│   └── starbucks_knowledge.jsonl # 공식 자료 전체 수집 결과
├── src/
│   ├── collect_starbucks_data.py # 공식 메뉴·FAQ 수집 (API 키 불필요)
│   ├── ingest.py             # Loader → Splitter → Embedding → Vector Store
│   ├── offline_demo.py       # API 키 없는 로컬 BM25 근거 검색
│   ├── retrieve.py           # Baseline / Hybrid(시도) / Reranker(최종)
│   ├── rag.py                # Prompt → Chat Model, 근거 없음 처리
│   ├── evaluate.py           # 고정 질문셋, Retrieval·Generation 평가, 결과 기록
│   └── capstone_compare.py   # 실행 진입점 (교재 TODO 1~7 순서)
└── results/
    ├── starbucks_design.md       # 현재 도메인 설계와 공식 출처
    ├── starbucks_evaluation.md   # 현재 질문셋과 안전성 판정 기준
    └── run_result.md             # 실행 시 자동 생성되는 결과
```

## 4. 설치 방법

공통 사전 준비: Python 3.11, [uv](https://docs.astral.sh/uv/). 데이터 수집과 오프라인 검색에는 OpenAI API Key가 필요하지 않습니다.

macOS / Linux:

```bash
git clone https://github.com/gitakchun/starbucks-menu-rag.git
cd starbucks-menu-rag
uv sync
# 전체 API 평가에만 필요: cp .env.example .env 후 OPENAI_API_KEY 입력
```

Windows PowerShell:

```powershell
git clone https://github.com/gitakchun/starbucks-menu-rag.git
cd starbucks-menu-rag
uv sync
# 전체 API 평가에만 필요: Copy-Item .env.example .env 후 OPENAI_API_KEY 입력
```


## 5. 실행 방법

공식 페이지에서 전체 데이터셋을 갱신합니다. 이 단계에는 API 키가 필요하지 않습니다.

```bash
uv run python src/collect_starbucks_data.py --delay 0.25
```

수집기만 빠르게 확인할 때는 메뉴 두 개로 시험할 수 있습니다.

```bash
uv run python src/collect_starbucks_data.py --limit 2 --skip-faq --delay 0 --output /tmp/starbucks-smoke.jsonl
```

전체 RAG 임베딩·생성 평가에는 `.env`의 `OPENAI_API_KEY`가 필요합니다.

```bash
uv run python src/capstone_compare.py
```

API 키 없이 전체 데이터 로딩·청킹·검색 근거를 확인하려면 로컬 BM25 데모를 실행합니다. 이 모드는 답변을 생성하지 않고 검색된 원문과 출처만 보여줍니다.

```bash
uv run python src/offline_demo.py --question "우유 알레르기인데 카페 라떼에 우유가 들어가나요?"
```

실행하면 다음 순서로 진행됩니다.

1. Chunk 목록 출력 (개수, 길이, metadata 확인)
2. Baseline / Hybrid / Reranker 각각 10개 질문의 Top-3 검색 결과, HIT/MISS, RR 출력 (Reranker는 관련도 점수 포함)
3. Baseline과 Reranker의 답변 생성과 LLM Judge 평가
4. 지표 비교와 최종 답변 예시 출력
5. `results/run_result.md` 갱신

`gpt-4o-mini` 호출은 실행 1회에 약 102번입니다 (Rerank 60회, 답변·Judge 40회, 예시 답변 2회). Embedding 호출은 별도입니다.

## 6. 평가 질문

| ID | 유형 | 질문 |
|---|---|---|
| Q1 | 알레르기 직설 질문 | 우유 알레르기 있는데 카페 라떼 마셔도 돼요? |
| Q2–Q3 | 카페인 비교·표현 바꾸기 | 카페 아메리카노랑 라떼 중에 카페인 적은 건 뭐예요? / 식물성 음료 들어가는 커피 있어요? |
| Q4–Q6 | 리워드·영양·복합 질문 | 별 쌓으려면 어떻게 결제해요? / 톨 라떼 칼로리는? / 오트 콜드 브루 카페인과 당류는? |
| Q7 | FAQ 근거 질문 | 신용카드로 결제했는데 별 적립이 안 된 이유 |
| Q8–Q10 | 거절 검증 | 미표시 알레르기 성분, 교차접촉 안전성, 매장별 실시간 판매 여부 |

## 7. 현재 결과

이 데이터셋에 대한 비교 평가 결과는 아직 측정 전입니다. 실행 후 `results/run_result.md`가 생성됩니다.

## 8. 한계와 추가 개선 방향

- **데이터 최신성:** 시즌 메뉴 및 영양 정보가 바뀔 수 있으므로 출처와 확인일을 갱신해야 합니다.
- **알레르기 안전:** 웹페이지의 성분 표시만으로 교차접촉이나 개인별 안전성을 보장할 수 없습니다.
- **Context 노이즈:** Reranker 점수가 10점 이하인 Chunk도 Top-3를 채웁니다. 관련도 임계값으로 걸러낼 수 있습니다.
- **비용·지연:** 질문마다 LLM 호출이 6회 늘어납니다. Cross-Encoder 전용 Reranker 모델로 바꿀 수 있습니다.
- **공식 FAQ 범위:** 답변 원문을 확인할 수 없는 FAQ 세부 규정과 개인별 주문 상태는 답변 대상에서 제외했습니다.
- **평가 신뢰도:** 질문셋 10개는 초기 회귀 검증용입니다. 실제 고객 표현으로 별도 검증셋을 확장해야 합니다.
