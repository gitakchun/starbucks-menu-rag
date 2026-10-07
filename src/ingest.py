"""PREPROCESSING: 문서 로드 → Chunk 분할 → Embedding → Vector Store."""

from pathlib import Path
import hashlib
import json
import pickle
import time

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMBEDDING_MODEL = "gemini-embedding-001"
CHUNK_SIZE = 450
CHUNK_OVERLAP = 40
# Gemini 무료 등급은 Embedding 요청이 분당 100건으로 제한됩니다.
EMBED_BATCH_SIZE = 90
EMBED_BATCH_INTERVAL = 61
EMBED_CACHE_PATH = Path(".cache") / f"embeddings_{EMBEDDING_MODEL}.pkl"


def load_documents(file_path: str):
    """UTF-8 텍스트 또는 수집한 JSONL 지식 레코드를 읽습니다."""

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

    if path.suffix == ".jsonl":
        docs = []
        with path.open(encoding="utf-8") as source_file:
            for line_number, line in enumerate(source_file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"잘못된 JSONL 레코드: {path}:{line_number}") from error

                if record.get("record_type") == "menu":
                    item_type = "음료" if record.get("item_type") == "drink" else "푸드"
                    title = f"[{item_type} / {record.get('category', '')}] {record.get('name', '')}"
                    lines = [title, record.get("description", "")]
                    properties = record.get("properties", {})
                    if properties:
                        nutrition = ", ".join(
                            f"{name}: {value}"
                            for name, value in properties.items()
                            if value and name not in {"상품유형", "영문명"}
                        )
                        if nutrition:
                            lines.append(f"규격·영양 정보: {nutrition}")
                    if record.get("allergen_status") == "listed":
                        lines.append(f"공식 알레르기 유발요인 표시: {record.get('allergens')}")
                    elif record.get("allergen_status") == "not_shown":
                        lines.append("공식 상세 페이지에 알레르기 유발요인 표시 없음. 무알레르기 또는 안전을 뜻하지 않음.")
                    else:
                        lines.append("알레르기 표기 상태를 확인하지 못함. 안전 여부를 추정하지 말 것.")
                    item_id = record.get("product_code", "")
                elif record.get("record_type") == "faq":
                    question = record.get("question", "")
                    title = f"[FAQ / {record.get('category', '')}] 질문: {question}"
                    lines = [title, f"공식 답변: {record.get('answer', '')}"]
                    item_id = record.get("faq_id", "")
                else:
                    raise ValueError(f"지원하지 않는 JSONL 레코드 유형: {path}:{line_number}")

                docs.append(Document(
                    page_content="\n".join(value for value in lines if value),
                    metadata={
                        "source": record.get("source_url", ""),
                        "record_type": record["record_type"],
                        "item_id": str(item_id),
                        "name": record.get("name", ""),
                        "category": record.get("category", ""),
                        "allergens": record.get("allergens") or "",
                        "collected_at": record.get("collected_at", ""),
                        "allergen_status": record.get("allergen_status", ""),
                        "question": record.get("question", ""),
                    },
                ))
        return docs

    return TextLoader(str(path), encoding="utf-8").load()


def split_documents(docs):
    """문단(빈 줄) 단위를 우선으로 Chunk를 나누고 metadata를 붙입니다."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    # 메뉴 레코드는 쪼개지 않습니다. 메뉴명과 알레르기 표시가 다른 Chunk로
    # 갈라지면 다른 메뉴의 알레르기 정보로 답할 수 있기 때문입니다.
    chunks = []
    for doc in docs:
        if doc.metadata.get("record_type") == "menu":
            chunks.append(doc)
        else:
            chunks.extend(splitter.split_documents([doc]))

    # 검색 결과를 비교할 때 어떤 Chunk가 나왔는지 번호로 추적합니다.
    for i, chunk in enumerate(chunks, start=1):
        chunk.metadata["chunk_id"] = i
        first_line = chunk.page_content.splitlines()[0]
        chunk.metadata["section"] = chunk.metadata.get("category") or (
            first_line if first_line.startswith("[") else ""
        )
        question = chunk.metadata.get("question", "")
        if question and question not in chunk.page_content:
            chunk.page_content = f"[FAQ 질문] {question}\n{chunk.page_content}"

    return chunks


def embed_with_cache(texts: list[str], embeddings) -> list[list[float]]:
    """분당 한도에 맞춰 나눠 Embedding하고, 결과를 디스크에 저장해 재실행 때 재사용합니다."""

    cache = pickle.loads(EMBED_CACHE_PATH.read_bytes()) if EMBED_CACHE_PATH.exists() else {}
    keys = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts]
    missing = list({key: text for key, text in zip(keys, texts) if key not in cache}.items())

    for start in range(0, len(missing), EMBED_BATCH_SIZE):
        if start:
            time.sleep(EMBED_BATCH_INTERVAL)
        batch = missing[start:start + EMBED_BATCH_SIZE]
        vectors = embeddings.embed_documents([text for _, text in batch])
        cache.update((key, vector) for (key, _), vector in zip(batch, vectors))
        EMBED_CACHE_PATH.parent.mkdir(exist_ok=True)
        EMBED_CACHE_PATH.write_bytes(pickle.dumps(cache))
        print(f"Embedding {min(start + EMBED_BATCH_SIZE, len(missing))}/{len(missing)}")

    return [cache[key] for key in keys]


def build_vectorstore(chunks):
    """Chunk를 Embedding하여 FAISS Vector Store를 만듭니다."""

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    texts = [chunk.page_content for chunk in chunks]
    return FAISS.from_embeddings(
        list(zip(texts, embed_with_cache(texts, embeddings))),
        embeddings,
        metadatas=[chunk.metadata for chunk in chunks],
    )
