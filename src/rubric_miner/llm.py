from __future__ import annotations

import json
import os
import re
from urllib.parse import urlparse
from typing import Any, List, Mapping, Optional, Sequence, TYPE_CHECKING

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

if TYPE_CHECKING:  # pragma: no cover
    from openai import AsyncOpenAI
else:
    AsyncOpenAI = Any


def build_client(
    *,
    api_key_env: str = "OPENAI_API_KEY",
    base_url: Optional[str] = None,
) -> AsyncOpenAI:
    try:
        from openai import AsyncOpenAI as OpenAIAsyncClient
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency: openai. Install runtime deps, for example: "
            "pip install openai tenacity rich pydantic"
        ) from exc

    kwargs = {"api_key": os.getenv(api_key_env, "local")}
    resolved_base_url = base_url or os.getenv("OPENAI_BASE_URL")
    if resolved_base_url:
        kwargs["base_url"] = resolved_base_url
        if _is_local_base_url(resolved_base_url):
            try:
                import httpx
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("Missing dependency: httpx. It is required by openai for local proxy bypass.") from exc
            _ensure_no_proxy_for_localhost()
            kwargs["http_client"] = httpx.AsyncClient(trust_env=False)
    return OpenAIAsyncClient(**kwargs)


def _is_local_base_url(base_url: str) -> bool:
    host = urlparse(base_url).hostname
    return host in {"localhost", "127.0.0.1", "0.0.0.0", "::1"}


def _ensure_no_proxy_for_localhost() -> None:
    local_hosts = ["localhost", "127.0.0.1", "0.0.0.0", "::1"]
    for key in ("NO_PROXY", "no_proxy"):
        existing = [part.strip() for part in os.getenv(key, "").split(",") if part.strip()]
        merged = existing + [host for host in local_hosts if host not in existing]
        os.environ[key] = ",".join(merged)


def extract_json_array(text: str) -> List[Any]:
    """Extract a strict JSON array from noisy LLM output."""

    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL | re.I)
    if fenced:
        cleaned = fenced.group(1).strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            for key in ("rubrics", "items", "signals", "discriminative_signals", "data"):
                value = parsed.get(key)
                if isinstance(value, list):
                    return value
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\[", cleaned):
        try:
            parsed, _ = decoder.raw_decode(cleaned[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            return parsed
    raise ValueError("LLM output does not contain a valid JSON array")


@retry(
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
async def async_llm_call(
    client: AsyncOpenAI,
    model: str,
    messages: Sequence[Mapping[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> str:
    response = await client.chat.completions.create(
        model=model,
        messages=list(messages),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError(f"empty response from model {model}")
    return content


async def llm_json_array(
    client: AsyncOpenAI,
    model: str,
    messages: Sequence[Mapping[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> List[Any]:
    content = await async_llm_call(client, model, messages, temperature=temperature, max_tokens=max_tokens)
    try:
        return extract_json_array(content)
    except Exception as exc:
        excerpt = re.sub(r"\s+", " ", content).strip()[:500]
        raise ValueError(f"{exc}; output_excerpt={excerpt}") from exc


@retry(
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
async def async_embedding_call(
    client: AsyncOpenAI,
    model: str,
    text: str,
) -> List[float]:
    response = await client.embeddings.create(model=model, input=text)
    embedding = response.data[0].embedding
    if not embedding:
        raise ValueError(f"empty embedding from model {model}")
    return list(embedding)


@retry(
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
async def async_embedding_batch_call(
    client: AsyncOpenAI,
    model: str,
    texts: Sequence[str],
) -> List[List[float]]:
    response = await client.embeddings.create(model=model, input=list(texts))
    data = sorted(response.data, key=lambda item: item.index)
    embeddings = [list(item.embedding) for item in data]
    if len(embeddings) != len(texts):
        raise ValueError(f"embedding count mismatch from model {model}: {len(embeddings)} != {len(texts)}")
    if any(not embedding for embedding in embeddings):
        raise ValueError(f"empty embedding from model {model}")
    return embeddings
