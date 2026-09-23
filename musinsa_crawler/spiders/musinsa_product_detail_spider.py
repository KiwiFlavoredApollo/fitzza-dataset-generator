import json
import sys
from pathlib import Path
from typing import Any, AsyncIterator, Generator, AsyncGenerator
from urllib.parse import urlparse

import scrapy
from bs4 import BeautifulSoup
from scrapy import Request
from scrapy.http import Response

from camel_to_snake import camel_to_snake, convert_keys


class MusinsaProductDetailSpider(scrapy.Spider):
    DETAIL_BASE_URL = "https://goods-detail.musinsa.com/api2/goods"
    IMAGE_BASE_URL = "https://image.msscdn.net/thumbnails"

    name = "musinsa_product_detail_spider"

    """
    ## 같은 종류의 옷이라도 색이 다르다면 다른 상품 아이디를 부여합니다.
    - 모두의 에센셜 블루종 자켓 체크 - https://www.musinsa.com/products/7161041
    - 모두의 에센셜 블루종 자켓 스웨이드 - https://www.musinsa.com/products/7291736
    
    ## 색이 달라도 같은 종류의 옷이면 같은 상품 아이디를 부여합니다.
    - 워시드 배럴 다트 코튼 팬츠 4 Color - https://www.musinsa.com/products/7122619 
    
    
    ## 상세정보
    - https://goods-detail.musinsa.com/api2/goods/7122619
    
    ## 태그
    - https://goods-detail.musinsa.com/api2/goods/7122619/tags
    
    ## 문의
    https://goods-detail.musinsa.com/api2/goods/7122619/question-and-answer?isExceptedSecret=false
    
    ## 리뷰
    https://goods.musinsa.com/api2/review/v1/view/list?page=0&pageSize=10&goodsNo=7122619&sort=up_cnt_desc&selectedSimilarNo=7122619&myFilter=false&hasPhoto=false&isExperience=false
    
    ## 특이사항
    대문자, 숫자, 언더스코어, 하이픈 만으로 구성된 키는 스네이크케이스로 변환하지 않습니다.
    - feature_flags
    - slow_rollout_flags
    """

    def __init__(self, category: object, input: Path, output: Path, **kwargs: Any):
        super().__init__(**kwargs)
        self.category = category
        self.input = input
        self.output = output

    async def start(self, **kwargs: Any) -> AsyncIterator[Any]:
        filename = self.input / "product_list" / f"{self.category["name"]}.jsonl"
        products = list()

        with open(filename, "r", encoding="UTF-8") as file:
            for line in file:
                products.append(json.loads(line))

        for product in products:
            url: str = f"{self.DETAIL_BASE_URL}/{product['goods_no']}"

            yield scrapy.Request(url)

    def parse(self, response: Response, **kwargs: Any) -> Any:
        loaded = json.loads(response.body)
        data = loaded["data"]
        data = convert_keys(data)

        output: Path = self.output / "product_detail" / self.category["name"]

        # yield from self.download_thumbnail_images(data, output)

        # yield from self.download_goods_images(data, output)

        yield from self.download_content_images(data, output)

        # self.download_product_details(data, output)

    def download_thumbnail_images(self, data: object, output: Path) -> Generator[Request, None, None]:
        url: str = f"{self.IMAGE_BASE_URL}{data["thumbnail_image_url"]}"

        yield scrapy.Request(
            url,
            callback=self.save_thumbnail_images,
            cb_kwargs={
                "data": data,
                "output": output
            },
        )

    def save_thumbnail_images(self, response: Response, data: object, output: Path) -> None:
        output = output / f"{data["goods_no"]}" / "thumbnail_images"
        output.mkdir(parents=True, exist_ok=True)

        filename: str = response.url.split("/")[-1]

        with open(output / filename, "wb") as file:
            file.write(response.body)

    def download_goods_images(self, data: object, output: Path) -> Generator[Request, None, None]:
        for image in data["goods_images"]:
            url: str = f"{self.IMAGE_BASE_URL}{image["image_url"]}"

            yield scrapy.Request(
                url,
                callback=self.save_goods_images,
                cb_kwargs={
                    "data": data,
                    "output": output
                },
            )

    def save_goods_images(self, response: Response, data: object, output: Path) -> None:
        output = output / f"{data["goods_no"]}" / "goods_images"
        output.mkdir(parents=True, exist_ok=True)

        filename: str = response.url.split("/")[-1]

        with open(output / filename, "wb") as file:
            file.write(response.body)

    def download_content_images(self, data: object, output: Path) -> Generator[Request, None, None]:
        for url in self.get_content_image_urls(data):
            yield scrapy.Request(
                url,
                callback=self.save_content_images,
                cb_kwargs={
                    "data": data,
                    "output": output
                },
            )

    def get_content_image_urls(self, data: object) -> list[str]:
        soup = BeautifulSoup(data["goods_contents"], "html.parser")

        return [
            self.normalize_image_url(img["src"])
            for img in soup.find_all("img")
            if img.get("src")
        ]

    def normalize_image_url(self, url: str) -> str:
        if url.startswith("//"):
            return "https:" + url

        return url

    def save_content_images(self, response: Response, data: object, output: Path) -> None:
        try:
            output = output / f"{data["goods_no"]}" / "content_images"
            output.mkdir(parents=True, exist_ok=True)

            filename = Path(urlparse(response.url).path).name

            with open(output / filename, "wb") as file:
                file.write(response.body)

        except (ValueError, OSError) as e:
            print(f"본문 이미지 다운로드 실패: {data["goods_no"]}", file=sys.stderr)

    def download_product_details(self, data: object, output: Path) -> None:
        output = output / f"{data["goods_no"]}"
        output.mkdir(parents=True, exist_ok=True)

        filename = f"{data["goods_no"]}.json"

        with open(output / filename, "w", encoding="UTF-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
