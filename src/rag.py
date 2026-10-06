"""RUNTIME 생성: 검색된 Document를 Context로 묶어 Chat Model로 답변합니다."""

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 모듈 import 시점에 Chat Model을 만들므로 여기서 .env를 먼저 읽습니다.
load_dotenv()

CHAT_MODEL = "gpt-4o-mini"
NO_ANSWER = "제공된 문서에서 확인할 수 없습니다."

prompt = ChatPromptTemplate.from_template("""
당신은 스타벅스 코리아의 공개 메뉴 및 서비스 정보를 안내하는 도우미입니다.
아래 문서에 명시된 사실만 근거로 답하고, 메뉴의 판매 여부나 매장별 운영 상황을 추측하지 마세요.
문서에 답이 없으면 "{no_answer}"라고 답하세요.
알레르기 질문에서는 문서에 명시된 알레르기 유발요인만 그대로 안내하세요.
알레르기 표시가 없다는 이유로 해당 성분이 없다고 말하거나, 알레르기 고객에게 안전하다고 보증하지 마세요.
교차접촉, 원재료의 세부 구성, 개인별 안전성이 문서에 없으면 답변을 보류하고 매장 파트너에게 확인하도록 안내하세요.
유당 불내증과 우유 알레르기를 같은 것으로 취급하지 말고, 의료적 판단이나 섭취 권고를 하지 마세요.
카페인 함량은 문서에 표시된 메뉴와 규격을 함께 말하세요. '저카페인'의 의학적 기준을 임의로 정하지 마세요.
질문이 여러 항목을 묻는다면 확인되는 사실만 답하고 나머지는 확인할 수 없다고 밝히세요.

[문서]
{context}

[질문]
{question}
""")

llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)


def format_docs(docs) -> str:
    return "\n\n".join(
        f"[문서 {i}]\n{doc.page_content}"
        for i, doc in enumerate(docs, start=1)
    )


def answer(question: str, docs) -> str:
    """이미 검색된 docs를 Context로 사용해 답변을 생성합니다."""

    messages = prompt.invoke({
        "no_answer": NO_ANSWER,
        "context": format_docs(docs),
        "question": question,
    })
    return llm.invoke(messages).content.strip()


def answer_with_retriever(question: str, retriever) -> str:
    """질문 → 검색 → Context → 답변 전체 흐름입니다."""

    return answer(question, retriever.invoke(question))
