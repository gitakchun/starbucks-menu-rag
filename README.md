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
starbucks_knowledge.jsonl → JSONL Documents → RecursiveCharacterTextSplitter(450/40) → GoogleGenerativeAIEmbeddings → FAISS

[RUNTIME]
Question → Retriever (아래 세 방식 비교) → Top-3 → Context → Prompt → gemini-3.5-flash-lite → Answer
```

같은 질문셋으로 세 가지 검색 방식을 비교합니다.

- Baseline: Dense Similarity Retriever (k=3)
- Hybrid: Dense + BM25 (RRF, 가중치 0.5/0.5) → Top-3
- Reranker: Dense 후보 6개 → LLM 관련도(0~100) Rerank → Top-3

이 데이터셋의 첫 측정에서는 Baseline이 가장 높았고, Hybrid와 Reranker는 Baseline을 넘지 못했습니다 (7장). 최종 채택 방식은 검증셋을 늘린 뒤 정합니다.

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
│   └── capstone_compare.py   # 실행 진입점
└── results/
    ├── starbucks_design.md       # 현재 도메인 설계와 공식 출처
    ├── starbucks_evaluation.md   # 현재 질문셋과 안전성 판정 기준
    └── run_result.md             # 실행 시 자동 생성되는 결과
```

## 4. 설치 방법

공통 사전 준비: Python 3.11, [uv](https://docs.astral.sh/uv/). 데이터 수집과 오프라인 검색에는 Google API Key가 필요하지 않습니다.

macOS / Linux:

```bash
git clone https://github.com/gitakchun/starbucks-menu-rag.git
cd starbucks-menu-rag
uv sync
# 전체 API 평가에만 필요: cp .env.example .env 후 GOOGLE_API_KEY 입력
```

Windows PowerShell:

```powershell
git clone https://github.com/gitakchun/starbucks-menu-rag.git
cd starbucks-menu-rag
uv sync
# 전체 API 평가에만 필요: Copy-Item .env.example .env 후 GOOGLE_API_KEY 입력
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

전체 RAG 임베딩·생성 평가에는 `.env`의 `GOOGLE_API_KEY`가 필요합니다.

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

`gemini-3.5-flash-lite` 호출은 실행 1회에 약 102번입니다 (Rerank 60회, 답변·Judge 40회, 예시 답변 2회). 무료 등급 한도에 맞춰 Chat은 분당 10건, Embedding은 61초마다 90건씩 보내므로 첫 실행은 20분쯤 걸립니다. Embedding 결과는 `.cache/`에 저장되어 재실행 때 다시 호출하지 않습니다.

## 6. 평가 질문

| ID | 유형 | 질문 |
|---|---|---|
| Q1 | 알레르기 직설 질문 | 우유 알레르기 있는데 카페 라떼 마셔도 돼요? |
| Q2–Q3 | 카페인 비교·표현 바꾸기 | 카페 아메리카노랑 라떼 중에 카페인 적은 건 뭐예요? / 식물성 음료 들어가는 커피 있어요? |
| Q4–Q6 | 리워드·영양·복합 질문 | 별 쌓으려면 어떻게 결제해요? / 톨 라떼 칼로리는? / 오트 콜드 브루 카페인과 당류는? |
| Q7 | FAQ 근거 질문 | 신용카드로 결제했는데 별 적립이 안 된 이유 |
| Q8–Q10 | 거절 검증 | 미표시 알레르기 성분, 교차접촉 안전성, 매장별 실시간 판매 여부 |

## 7. 현재 결과

2026-10-07 측정 결과입니다. 조건: `gemini-embedding-001`, `gemini-3.5-flash-lite`(답변·Rerank·Judge 공통, temperature=0), Top-3, 답변 가능 질문 7개(Q1~Q7). 질문별 상세는 `results/run_result.md`에 있습니다.

| 지표 | Baseline Dense | Hybrid | Reranker |
|---|---|---|---|
| Retrieval Hit Rate (7문항) | 1.000 | 1.000 | 0.857 |
| Retrieval MRR (7문항) | 0.833 | 0.810 | 0.786 |
| Generation 근거성 평균 (1~5) | 5.00 | - | 5.00 |
| Generation 직접 답변 비율 | 0.714 | - | 0.571 |

- **Baseline이 이미 7문항을 모두 맞혔습니다.** 개선안이 더 나아질 여지가 없는 질문셋이라, 방식 간 우열을 가리기에는 부족합니다.
- **Hybrid는 Baseline과 Hit Rate가 같고 MRR만 낮습니다.** 차이는 Q7 하나에서 났습니다. BM25가 "별 적립"이 겹치는 다른 카드 FAQ를 2위로 올려, 정답 근거가 3위로 밀렸습니다.
- **Reranker는 Q2를 놓쳤습니다.** 카페 아메리카노 Chunk를 후보에서 떨어뜨리고 아이스 카페 라떼를 올려, 카페인 비교에 필요한 150mg 근거가 Top-3에서 빠졌습니다.
- **Q8·Q9 답변은 안전 원칙을 지켰습니다.** 두 방식 모두 "표시 없음은 무알레르기나 안전을 뜻하지 않는다"고 답했고, 안전을 보증하지 않았습니다.

해석할 때 주의할 점:

- **거절·직접 답변 지표는 그대로 믿기 어렵습니다.** 거절 여부를 "확인할 수 없" 문구 포함으로만 판정합니다. 그래서 Q8·Q9처럼 다른 표현으로 올바르게 보류한 답변은 거절 실패로 집계되고, Q3·Q5처럼 답을 한 뒤 "그 외는 확인할 수 없다"고 덧붙인 답변은 답하지 않은 것으로 집계됩니다. `run_result.md`의 "문서에 없는 질문 거절: 실패"는 이 판정 방식 때문입니다.
- **Reranker 결과는 lite 모델 기준입니다.** 무료 등급 한도 때문에 `gemini-3.5-flash-lite`로 측정했습니다. 더 큰 모델로 Rerank하면 결과가 달라질 수 있습니다.
- **1회 측정입니다.** 반복 실행에 따른 편차는 확인하지 않았습니다.

## 8. 한계와 추가 개선 방향

- **데이터 최신성:** 시즌 메뉴 및 영양 정보가 바뀔 수 있으므로 출처와 확인일을 갱신해야 합니다.
- **알레르기 안전:** 웹페이지의 성분 표시만으로 교차접촉이나 개인별 안전성을 보장할 수 없습니다.
- **Context 노이즈:** Reranker 점수가 10점 이하인 Chunk도 Top-3를 채웁니다. 관련도 임계값으로 걸러낼 수 있습니다.
- **비용·지연:** 질문마다 LLM 호출이 6회 늘어납니다. Cross-Encoder 전용 Reranker 모델로 바꿀 수 있습니다.
- **공식 FAQ 범위:** 답변 원문을 확인할 수 없는 FAQ 세부 규정과 개인별 주문 상태는 답변 대상에서 제외했습니다.
- **평가 신뢰도:** 질문셋 10개는 초기 회귀 검증용입니다. 실제 고객 표현으로 별도 검증셋을 확장해야 합니다.
