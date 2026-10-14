import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import Depends, FastAPI, Request
from pydantic import BaseModel

from law_links import Extractor, LawIndex

log = logging.getLogger("law_links")

ALIASES_PATH = os.environ.get("LAW_ALIASES_PATH", str(Path(__file__).resolve().parent / "law_aliases.json"))


class LawLink(BaseModel):
    law_id: Optional[int] = None
    article: Optional[str] = None
    point_article: Optional[str] = None
    subpoint_article: Optional[str] = None


class LinksResponse(BaseModel):
    links: List[LawLink]


class TextRequest(BaseModel):
    text: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: индекс законов строится один раз (~1-2 с)
    t0 = time.perf_counter()
    index = LawIndex.from_file(ALIASES_PATH)
    extractor = Extractor(index)
    extractor.extract("п. 1 ст. 1 НК РФ")  # прогрев кэшей
    app.state.extractor = extractor
    print(f"🚀 Сервис запускается... индекс законов: {len(index.laws)} шт., {time.perf_counter() - t0:.2f} с")
    yield
    # Shutdown
    del app.state.extractor
    print("🛑 Сервис завершается...")


def get_extractor(request: Request) -> Extractor:
    return request.app.state.extractor


app = FastAPI(
    title="Law Links Service",
    description="Cервис для выделения юридических ссылок из текста",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/detect")
async def get_law_links(
    data: TextRequest,
    extractor: Extractor = Depends(get_extractor),
) -> LinksResponse:
    """
    Принимает текст и возвращает список юридических ссылок
    """
    try:
        found = extractor.extract(data.text)
    except Exception:  # сервис не должен падать на странном тексте
        log.exception("extract failed")
        found = []
    links = [
        LawLink(law_id=law_id, article=article, point_article=point, subpoint_article=subpoint)
        for law_id, article, point, subpoint in found
    ]
    return LinksResponse(links=links)


@app.get("/health")
async def health_check():
    """
    Проверка состояния сервиса
    """
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8978)
