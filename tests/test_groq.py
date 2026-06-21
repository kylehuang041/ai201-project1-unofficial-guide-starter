import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from src.rag import _client  # noqa: E402


@pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set; skipping Groq client smoke test.",
)
def test_groq_client_smoke():
    client = _client()
    assert hasattr(client, "chat")
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=10,
        messages=[{"role": "user", "content": "Reply with the word OK."}],
    )
    raw_text = resp.choices[0].message.content.strip()
    text = raw_text.lower()
    print(f"Groq response: {raw_text}")
    assert "ok" in text

