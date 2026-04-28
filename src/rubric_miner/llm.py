from __future__ import annotations

import json
import os
import re
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
    return OpenAIAsyncClient(**kwargs)


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
) -> str:
    response = await client.chat.completions.create(
        model=model,
        messages=list(messages),
        temperature=temperature,
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
) -> List[Any]:
    content = await async_llm_call(client, model, messages, temperature=temperature)
    return extract_json_array(content)


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
