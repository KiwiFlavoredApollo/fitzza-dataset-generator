import json
from pathlib import Path
from typing import Self, Any
from urllib.parse import urlencode

import scrapy
from scrapy.http import Response

from camel_to_snake import convert_keys


class MusinsaProductListSpider(scrapy.Spider):
    BASE_URL: str = "https://api.musinsa.com/api2/dp/v2/plp/goods"

    """
    ## 예시 URL
    https://api.musinsa.com/api2/dp/v2/plp/goods?gf=M&sortCode=POPULAR&size=60&caller=CATEGORY&page=1&hmacId=1825e8016080aa31f6813eb000ece8009c81d2052350e598da7509934cf551ad&category=001

    ## 특이사항
    - 웹브라우저 개발자도구의 네트워크 탭을 조사하면서 API URL을 얻었습니다. 
    - data.pagination.nextPageUrl을 통해 다음 페이지를 가져올 수 있습니다.
    - size를 예를 들어 200로 크게 설정하면 오류가 발생합니다.
    - hmacId가 페이지마다 다릅니다.
    """
    PARAMETERS: dict = {
        "gf": "M",
        "sortCode": "POPULAR",
        "size": 100,
        "caller": "CATEGORY",
        "page": 1,
        "hmacId": "1825e8016080aa31f6813eb000ece8009c81d2052350e598da7509934cf551ad",
    }

    name = "musinsa_product_list_spider"

    def __init__(self, category: object, output: Path,  **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.category = category
        self.output = output

    async def start(self):
        url = (
            f"{self.BASE_URL}"
            f"?{urlencode(self.PARAMETERS)}"
            f"&category={self.category["code"]}"
        )

        yield scrapy.Request(url)

    def parse(self, response: Response, **kwargs: Any) -> Any:
        try:
            loaded = json.loads(response.body)
            products = loaded["data"]["list"]
            products = convert_keys(products)

            output = self.output / "product_list"
            output.mkdir(parents=True, exist_ok=True)

            with open(output / f"{self.category["name"]}.jsonl", "w", encoding="UTF-8") as file:
                for product in products:
                    file.write(json.dumps(product, ensure_ascii=False) + "\n")

            next_page_url: str = loaded["data"]["pagination"]["nextPageUrl"]

        except KeyError:
            pass
