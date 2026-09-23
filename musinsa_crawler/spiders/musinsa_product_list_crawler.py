from typing import Self
from urllib.parse import urlencode

import scrapy
from scrapy.http import Response

from musinsa_crawler.musinsa.categories import *


class MusinsaProductListSpider(scrapy.Spider):
    BASE_URL: str = "https://api.musinsa.com/api2/dp/v2/plp/goods"

    """
    ## 예시 URL
    https://api.musinsa.com/api2/dp/v2/plp/goods?gf=M&sortCode=POPULAR&size=60&caller=CATEGORY&page=1&hmacId=1825e8016080aa31f6813eb000ece8009c81d2052350e598da7509934cf551ad&category=001

    ## 특이사항
    - data.pagination.nextPageUrl을 통해 다음 페이지를 가져올 수 있습니다.
    - size를 예를 들어 200로 크게 설정하면 안됩니다.
    - hmacId가 페이지 마다 다릅니다.
    """
    PARAMETERS: dict = {
        "gf": "M",
        "sortCode": "POPULAR",
        "size": 100,
        "caller": "CATEGORY",
        "page": 1,
        "hmacId": "1825e8016080aa31f6813eb000ece8009c81d2052350e598da7509934cf551ad",
    }

    name = "musinsa_product_list_crawler"

    custom_settings = {
        "USER_AGENT": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/140.0.0.0 Safari/537.36"
        ),
    }

    async def start(self):
        for category in self.get_categories():
            url = (
                f"{self.BASE_URL}"
                f"?{urlencode(self.PARAMETERS)}"
                f"&category={category}"
            )

            yield scrapy.Request(url)

    def parse(self, response: Response, **kwargs) -> Self:
        # products = response.css('a[class^="GoodsItem"]')
        #
        # for product in products[:self.COUNTS]:
        #     product.css('::attr(href)').get()
        pass

    def get_categories(self) -> list[str]:
        return [TOPS, OUTERWEAR, BOTTOMS, HEADWEAR, ACTIVEWEAR]
