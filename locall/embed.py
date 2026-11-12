"""EmbeddingGemma 2 래퍼. 벡터는 dim 차원으로 자르고 정규화해서 돌려준다."""
import os
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(BASE / "models"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np

QUERY_PREFIX = "task: search result | query: "


SPECIAL_TOKENS = (
    ("<|image|>", "<image>"),
    ("<|video|>", "<video>"),
    ("<|audio|>", "<audio>"),
    ("<|image>", "<image>"),
    ("<image|>", "<image>"),
    ("<|audio>", "<audio>"),
    ("<audio|>", "<audio>"),
)


def _sanitize(text: str) -> str:
    text = text.encode("utf-16", "surrogatepass").decode("utf-16", "ignore").encode("utf-8", "ignore").decode("utf-8")
    for token, repl in SPECIAL_TOKENS:
        if token in text:
            text = text.replace(token, repl)
    return text


def doc_text(title: str, content: str) -> str:
    return f"title: {title or 'none'} | text: {content}"


class Embedder:
    def __init__(self, cfg: dict, images: bool = True):
        import torch
        from sentence_transformers import SentenceTransformer

        self.dim = cfg["dim"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # float16은 이 모델에서 조용히 품질이 망가진다 → bfloat16 또는 float32만 사용
        use_bf16 = self.device == "cuda" and torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if use_bf16 else torch.float32
        config_kwargs = {"audio_config": None}
        if not images:
            config_kwargs["vision_config"] = None

        model_name = cfg["model"]
        cache_hub = BASE / "models" / "hub"
        model_dir_name = "models--" + model_name.replace("/", "--")
        if not (cache_hub / model_dir_name).exists():
            print(f"[모델] 로컬 저장소(models/)에 모델({model_name})이 없어 다운로드합니다…", flush=True)

        self.model = SentenceTransformer(
            model_name,
            device=self.device,
            model_kwargs={"torch_dtype": dtype},
            config_kwargs=config_kwargs,
        )

    def _finish(self, emb) -> np.ndarray:
        emb = np.asarray(emb, dtype=np.float32)[:, : self.dim]
        emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9)
        return emb

    def docs(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """texts는 doc_text()로 만든 문자열. 저장용 float16."""
        texts = [_sanitize(t) for t in texts]
        emb = self.model.encode(texts, prompt="", batch_size=batch_size)
        return self._finish(emb).astype(np.float16)

    def images(self, pil_images: list, batch_size: int = 8) -> np.ndarray:
        emb = self.model.encode(pil_images, batch_size=batch_size)
        return self._finish(emb).astype(np.float16)

    def query(self, text: str) -> np.ndarray:
        emb = self.model.encode([QUERY_PREFIX + _sanitize(text)], prompt="")
        return self._finish(emb)[0]
