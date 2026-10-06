"""PREPROCESSING: 문서 로드 → Chunk 분할 → Embedding → Vector Store."""

from pathlib import Path
import json

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 450
CHUNK_OVERLAP = 40


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
                        "category": record.get("category", ""),
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
    chunks = splitter.split_documents(docs)

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


def build_vectorstore(chunks):
    """Chunk를 Embedding하여 FAISS Vector Store를 만듭니다."""

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return FAISS.from_documents(chunks, embeddings)
