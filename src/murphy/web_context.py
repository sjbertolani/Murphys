from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Protocol
from urllib.parse import urlencode
from urllib.request import urlopen
from xml.etree import ElementTree


@dataclass(frozen=True)
class NewsContextItem:
    ticker: str
    title: str
    link: str | None
    published_at: datetime | None
    source: str
    summary: str | None = None

    def as_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "title": self.title,
            "link": self.link,
            "published_at": None if self.published_at is None else self.published_at.isoformat(),
            "source": self.source,
            "summary": self.summary,
        }


class NewsContextProvider(Protocol):
    name: str

    def fetch(self, tickers: list[str], limit_per_ticker: int = 5) -> list[NewsContextItem]: ...


class YahooFinanceNewsProvider:
    name = "yahoo_finance_news"

    def fetch(self, tickers: list[str], limit_per_ticker: int = 5) -> list[NewsContextItem]:
        items: list[NewsContextItem] = []
        for ticker in sorted({ticker.strip().upper() for ticker in tickers if ticker.strip()}):
            items.extend(self._fetch_one(ticker, limit_per_ticker=limit_per_ticker))
        return items

    def _fetch_one(self, ticker: str, limit_per_ticker: int) -> list[NewsContextItem]:
        query = urlencode({"s": ticker, "region": "US", "lang": "en-US"})
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?{query}"
        with urlopen(url, timeout=15) as response:
            payload = response.read()
        root = ElementTree.fromstring(payload)
        parsed: list[NewsContextItem] = []
        for item in root.findall("./channel/item")[:limit_per_ticker]:
            title = _text(item, "title")
            if not title:
                continue
            parsed.append(
                NewsContextItem(
                    ticker=ticker,
                    title=title,
                    link=_text(item, "link"),
                    published_at=_parse_published_at(_text(item, "pubDate")),
                    source="Yahoo Finance RSS",
                    summary=_text(item, "description"),
                )
            )
        return parsed


def news_items_payload(items: list[NewsContextItem]) -> dict:
    return {
        "items": [item.as_dict() for item in items],
        "counts": {
            "items": len(items),
            "tickers": len({item.ticker for item in items}),
        },
    }


def _text(element: ElementTree.Element, tag: str) -> str | None:
    found = element.find(tag)
    if found is None or found.text is None:
        return None
    text = found.text.strip()
    return text or None


def _parse_published_at(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
