from __future__ import annotations

from collections import Counter

from bs4 import BeautifulSoup


def extract_hrefs(html: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    return [a.get("href", "") for a in soup.find_all("a") if a.has_attr("href")]


def validate_hrefs(original_html: str, translated_html: str) -> tuple[bool, str]:
    before = extract_hrefs(original_html)
    after = extract_hrefs(translated_html)
    if before != after:
        return False, f"href mismatch: before={before} after={after}"
    return True, "ok"


def tag_multiset(html: str) -> Counter:
    soup = BeautifulSoup(html, "lxml")
    return Counter([tag.name for tag in soup.find_all()])


def attribute_multiset(html: str) -> Counter:
    soup = BeautifulSoup(html, "lxml")
    attrs: Counter = Counter()
    for tag in soup.find_all():
        for attr_name, attr_val in tag.attrs.items():
            attrs[(tag.name, attr_name, str(attr_val))] += 1
    return attrs


def validate_structure(original_html: str, translated_html: str) -> tuple[bool, str]:
    tags_ok = tag_multiset(original_html) == tag_multiset(translated_html)
    attrs_ok = attribute_multiset(original_html) == attribute_multiset(translated_html)
    if tags_ok and attrs_ok:
        return True, "ok"
    return False, "tag/attribute mismatch"
