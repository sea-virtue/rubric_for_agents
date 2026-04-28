from __future__ import annotations

import argparse
import time
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str = ""


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: float = 0.2
    max_tokens: int = Field(default=2048, alias="max_completion_tokens")

    class Config:
        allow_population_by_field_name = True
        extra = "allow"


class EmbeddingRequest(BaseModel):
    model: str
    input: Any

    class Config:
        extra = "allow"


class LocalModels:
    def __init__(
        self,
        *,
        chat_model: str,
        embedding_model: str,
        served_chat_model_name: str,
        served_embedding_model_name: str,
        device: str,
    ) -> None:
        self.chat_model_name = chat_model
        self.embedding_model_name = embedding_model
        self.served_chat_model_name = served_chat_model_name or chat_model
        self.served_embedding_model_name = served_embedding_model_name or embedding_model
        self.device = device
        self.tokenizer = None
        self.chat_model = None
        self.embedding_model = None

    def load(self) -> None:
        if self.chat_model_name:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.chat_model_name, trust_remote_code=True)
            self.chat_model = AutoModelForCausalLM.from_pretrained(
                self.chat_model_name,
                device_map=self.device,
                torch_dtype="auto",
                trust_remote_code=True,
            )
        if self.embedding_model_name:
            from sentence_transformers import SentenceTransformer

            if self.device == "auto":
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
            else:
                self.embedding_model = SentenceTransformer(self.embedding_model_name, device=self.device)

    def chat(self, request: ChatCompletionRequest) -> str:
        if self.chat_model is None or self.tokenizer is None:
            raise HTTPException(status_code=404, detail="chat model is not loaded")
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages) + "\nassistant:"
        inputs = self.tokenizer([text], return_tensors="pt").to(self.chat_model.device)

        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": int(request.max_tokens or 2048),
            "do_sample": request.temperature > 0,
            "temperature": max(float(request.temperature), 1e-5),
        }
        output_ids = self.chat_model.generate(**inputs, **generation_kwargs)
        generated = output_ids[0][inputs.input_ids.shape[-1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def embeddings(self, request: EmbeddingRequest) -> List[List[float]]:
        if self.embedding_model is None:
            raise HTTPException(status_code=404, detail="embedding model is not loaded")
        texts = request.input if isinstance(request.input, list) else [request.input]
        vectors = self.embedding_model.encode([str(text) for text in texts], normalize_embeddings=True)
        return [vector.astype(float).tolist() for vector in vectors]


def create_app(models: LocalModels) -> FastAPI:
    app = FastAPI(title="Minimal HF OpenAI-Compatible Server")

    @app.get("/v1/models")
    def list_models() -> Dict[str, Any]:
        data = []
        if models.chat_model_name:
            data.append({"id": models.served_chat_model_name, "object": "model"})
        if models.embedding_model_name:
            data.append({"id": models.served_embedding_model_name, "object": "model"})
        return {"object": "list", "data": data}

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest) -> Dict[str, Any]:
        content = models.chat(request)
        created = int(time.time())
        return {
            "id": f"chatcmpl-local-{created}",
            "object": "chat.completion",
            "created": created,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": content},
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    @app.post("/v1/embeddings")
    def embeddings(request: EmbeddingRequest) -> Dict[str, Any]:
        vectors = models.embeddings(request)
        return {
            "object": "list",
            "model": request.model,
            "data": [
                {"object": "embedding", "embedding": vector, "index": idx}
                for idx, vector in enumerate(vectors)
            ],
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        }

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal local HF OpenAI-compatible server")
    parser.add_argument("--chat-model", default="")
    parser.add_argument("--embedding-model", default="")
    parser.add_argument("--served-chat-model-name", default="")
    parser.add_argument("--served-embedding-model-name", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = LocalModels(
        chat_model=args.chat_model,
        embedding_model=args.embedding_model,
        served_chat_model_name=args.served_chat_model_name,
        served_embedding_model_name=args.served_embedding_model_name,
        device=args.device,
    )
    models.load()
    uvicorn.run(create_app(models), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
