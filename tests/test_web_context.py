from __future__ import annotations

from datetime import UTC, datetime

from murphy.web_context import NewsContextItem, news_items_payload


def test_news_items_payload() -> None:
    payload = news_items_payload(
        [
            NewsContextItem(
                ticker="AAPL",
                title="Apple headline",
                link="https://example.com/aapl",
                published_at=datetime(2026, 4, 27, 16, 0, tzinfo=UTC),
                source="test",
                summary="summary",
            )
        ]
    )

    assert payload["counts"] == {"items": 1, "tickers": 1}
    assert payload["items"][0]["ticker"] == "AAPL"
    assert payload["items"][0]["published_at"] == "2026-04-27T16:00:00+00:00"
