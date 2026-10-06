"""Collect public Starbucks Korea menu and FAQ facts into JSONL."""

import argparse
from datetime import datetime, timezone
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE_URL = "https://www.starbucks.co.kr"
USER_AGENT = "RAGCapstonePortfolio/1.0 (public menu data; low-rate collection)"
REQUEST_TIMEOUT = 25

MENU_CATEGORIES = [
    ("drink", "콜드 브루 커피", "W0000171"),
    ("drink", "브루드 커피", "W0000060"),
    ("drink", "에스프레소", "W0000003"),
    ("drink", "프라푸치노", "W0000004"),
    ("drink", "블렌디드", "W0000005"),
    ("drink", "스타벅스 리프레셔", "W0000422"),
    ("drink", "스타벅스 피지오", "W0000061"),
    ("drink", "티(티바나)", "W0000075"),
    ("drink", "기타 제조 음료", "W0000053"),
    ("drink", "스타벅스 주스(병음료)", "W0000062"),
    ("food", "브레드", "W0000013"),
    ("food", "케이크", "W0000032"),
    ("food", "샌드위치 & 샐러드", "W0000033"),
    ("food", "따뜻한 푸드", "W0000054"),
    ("food", "과일 & 요거트", "W0000055"),
    ("food", "스낵 & 미니 디저트", "W0000056"),
    ("food", "아이스크림", "W0000064"),
]

FAQ_CATEGORIES = {
    "회원",
    "결제/환불",
    "음료/푸드/상품",
    "매장",
    "딜리버스(배달)",
    "회원탈퇴",
    "이벤트/프로모션",
    "기타",
}


class JsonLdParser(HTMLParser):
    """Collect application/ld+json script contents from a product page."""

    def __init__(self):
        super().__init__()
        self._capture = False
        self._parts = []
        self.documents = []

    def handle_starttag(self, tag, attrs):
        if tag == "script" and dict(attrs).get("type") == "application/ld+json":
            self._capture = True
            self._parts = []

    def handle_data(self, data):
        if self._capture:
            self._parts.append(data)

    def handle_endtag(self, tag):
        if tag == "script" and self._capture:
            self.documents.append("".join(self._parts))
            self._capture = False


class PlainTextParser(HTMLParser):
    """Turn FAQ answer markup into readable plain text."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_data(self, data):
        value = " ".join(data.split())
        if value:
            self.parts.append(value)


def request_text(url: str, delay: float = 0.0, form: dict | None = None) -> str:
    if delay:
        time.sleep(delay)

    body = urlencode(form).encode("utf-8") if form is not None else None
    headers = {"User-Agent": USER_AGENT}
    if body is not None:
        headers["Content-Type"] = "application/x-www-form-urlencoded; charset=UTF-8"
    request = Request(url, data=body, headers=headers)

    for attempt in range(3):
        try:
            with urlopen(request, timeout=REQUEST_TIMEOUT) as response:
                return response.read().decode("utf-8", errors="replace")
        except (HTTPError, URLError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(1 + attempt)

    raise RuntimeError(f"요청 실패: {url}")


def request_json(url: str, delay: float = 0.0, form: dict | None = None):
    return json.loads(request_text(url, delay=delay, form=form))


def product_from_jsonld(page_html: str) -> dict | None:
    parser = JsonLdParser()
    parser.feed(page_html)

    for source in parser.documents:
        try:
            data = json.loads(source)
        except json.JSONDecodeError:
            continue

        candidates = data.get("@graph", [data]) if isinstance(data, dict) else []
        for candidate in candidates:
            product_type = candidate.get("@type")
            if product_type == "Product" or (
                isinstance(product_type, list) and "Product" in product_type
            ):
                return candidate
    return None


def parse_product_detail(page_html: str) -> dict | None:
    product = product_from_jsonld(page_html)
    if product is None:
        return None

    properties = {}
    for prop in product.get("additionalProperty", []):
        name = prop.get("name")
        if name:
            properties[name] = prop.get("value", "")

    allergens = str(properties.get("알레르기 유발요인", "")).strip()
    return {
        "english_name": properties.get("영문명", ""),
        "description": product.get("description", ""),
        "properties": properties,
        "allergen_status": "listed" if allergens else "not_shown",
        "allergens": allergens or None,
    }


def fetch_menu_records(limit: int | None, delay: float, collected_at: str) -> list[dict]:
    records = []
    seen = set()

    for item_type, category, category_code in MENU_CATEGORIES:
        menu_url = f"{BASE_URL}/upload/json/menu/{category_code}.js"
        payload = request_json(menu_url, delay=delay)
        for listing in payload.get("list", []):
            code = str(listing.get("product_CD", "")).strip()
            if not code or (item_type, code) in seen:
                continue
            seen.add((item_type, code))

            detail_path = "drink_view.do" if item_type == "drink" else "food_view.do"
            source_url = f"{BASE_URL}/menu/{detail_path}?product_cd={code}"
            detail_html = request_text(source_url, delay=delay)
            details = parse_product_detail(detail_html)
            if details is None:
                details = {
                    "english_name": "",
                    "description": "",
                    "properties": {},
                    "allergen_status": "not_parsed",
                    "allergens": None,
                }

            records.append({
                "record_type": "menu",
                "item_type": item_type,
                "product_code": code,
                "name": details["properties"].get("상품명", listing.get("product_NM", "")),
                "english_name": details["english_name"],
                "category": category,
                "description": details["description"] or listing.get("content", ""),
                "properties": details["properties"],
                "allergen_status": details["allergen_status"],
                "allergens": details["allergens"],
                "source_url": source_url,
                "collected_at": collected_at,
            })
            print(f"[menu {len(records)}] {category}: {listing.get('product_NM', code)}")

            if limit is not None and len(records) >= limit:
                return records

    return records


def plain_text(markup: str) -> str:
    parser = PlainTextParser()
    parser.feed(markup or "")
    return " ".join(parser.parts)


def fetch_faq_categories(delay: float) -> dict[str, str]:
    payload = request_json(
        f"{BASE_URL}/voc/getVocCauseCodeList.do",
        delay=delay,
        form={},
    )
    return {
        entry["code_NAME"]: str(entry["code"])
        for entry in payload.get("list", [])
        if entry.get("code_NAME") in FAQ_CATEGORIES and entry.get("code")
    }


def fetch_faq_records(delay: float, collected_at: str) -> list[dict]:
    categories = fetch_faq_categories(delay)
    missing = FAQ_CATEGORIES - categories.keys()
    if missing:
        raise RuntimeError(f"FAQ 분류 코드를 찾지 못했습니다: {sorted(missing)}")

    records = []
    seen = set()
    for category in sorted(FAQ_CATEGORIES):
        page = 1
        total = 0
        page_size = 0
        page_count = 1
        while page <= page_count:
            payload = request_json(
                f"{BASE_URL}/voc/getFaqList.do",
                delay=delay,
                form={
                    "CD_VOC_CAUSE_1": categories[category],
                    "CD_VOC_CAUSE_2": "",
                    "KEYWORD": "",
                    "pageIndex": page,
                    "ID_VOC_FAQ": "",
                },
            )
            page_records = payload.get("list", [])
            if not page_records:
                break
            if page == 1:
                total = int(payload.get("totCnt", 0) or 0)
                page_size = len(page_records)
                page_count = math.ceil(total / page_size) if page_size else 1

            for item in page_records:
                faq_id = str(item.get("id_VOC_FAQ", "")).strip()
                if not faq_id or faq_id in seen:
                    continue
                seen.add(faq_id)
                records.append({
                    "record_type": "faq",
                    "faq_id": faq_id,
                    "category": item.get("nm_VOC_CAUSE", category),
                    "question": plain_text(item.get("ds_QESTN", "")),
                    "answer": plain_text(item.get("ds_ANSWER", "")),
                    "source_url": f"{BASE_URL}/util/faq.do#faq-{faq_id}",
                    "collected_at": collected_at,
                })

            print(f"[faq] {category}: {min(page * page_size, total)}/{total}")
            page += 1

    return records


def main():
    parser = argparse.ArgumentParser(description="공식 공개 메뉴·FAQ 데이터를 JSONL로 수집합니다.")
    parser.add_argument("--output", type=Path, default=Path("data/starbucks_knowledge.jsonl"))
    parser.add_argument("--limit", type=int, help="메뉴 상세 수집 개수 제한(수집기 점검용)")
    parser.add_argument("--skip-faq", action="store_true", help="FAQ 수집 생략")
    parser.add_argument("--delay", type=float, default=0.25, help="요청 간 대기 시간(초)")
    args = parser.parse_args()

    if args.delay < 0:
        parser.error("--delay는 0 이상이어야 합니다.")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit은 1 이상이어야 합니다.")

    collected_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    menu_records = fetch_menu_records(args.limit, args.delay, collected_at)
    faq_records = [] if args.skip_faq else fetch_faq_records(args.delay, collected_at)
    records = menu_records + faq_records

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary_path.replace(args.output)

    not_parsed = sum(record.get("allergen_status") == "not_parsed" for record in menu_records)
    print(
        f"수집 완료: 메뉴 {len(menu_records)}개, FAQ {len(faq_records)}개, "
        f"알레르기 필드 파싱 확인 필요 {not_parsed}개 → {args.output}"
    )


if __name__ == "__main__":
    main()