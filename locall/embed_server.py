"""Rust 앱(locall.exe)이 띄우는 임베딩 작업자:  python -m locall.embed_server

색인을 만든 것과 똑같은 모델로 검색어/사진을 벡터로 바꿔 준다.
표준입력으로 JSON 한 줄을 받고, 표준출력으로 JSON 한 줄을 답한다.
  {"op": "query", "text": "..."}   → {"ok": true, "vec": [...]}
  {"op": "image", "path": "..."}   → {"ok": true, "vec": [...]}
표준입력이 닫히면(부모가 끝나면) 같이 끝난다.
"""
import json
import os
import sys


def _load_query_image(path: str):
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((896, 896))
        return im


def main():
    # 라이브러리가 stdout에 찍는 글자가 응답에 섞이지 않도록, 응답 전용 통로를 따로 떼어 둔다
    out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    def send(obj):
        out.write(json.dumps(obj).encode("utf-8") + b"\n")
        out.flush()

    try:
        from . import config

        cfg = config.load()
        cached = config.DEFAULT_MODELS / "hub" / ("models--" + cfg["model"].replace("/", "--"))
        if cached.exists():  # 이미 받아 둔 모델이면 네트워크 확인 없이 바로 연다
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from .embed import Embedder

        emb = Embedder(cfg, images=True)
        emb.query("예열")  # 첫 추론은 느리므로 미리 한 번
    except Exception as e:
        send({"ok": False, "error": f"{type(e).__name__}: {e}"})
        return
    send({"ok": True, "ready": True, "device": emb.device, "dim": emb.dim})

    for line in sys.stdin.buffer:
        try:
            req = json.loads(line)
            if req["op"] == "query":
                vec = emb.query(req["text"])
            elif req["op"] == "image":
                vec = emb.images([_load_query_image(req["path"])])[0]
            else:
                raise ValueError(f"알 수 없는 요청: {req['op']}")
            send({"ok": True, "vec": [float(x) for x in vec]})
        except Exception as e:
            send({"ok": False, "error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    main()
