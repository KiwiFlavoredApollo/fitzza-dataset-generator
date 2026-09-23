from typing import Self
from urllib.parse import urlencode

import scrapy
from scrapy.http import Response


class MusinsaCategoryCrawler(scrapy.Spider):
    COUNTS: int = 10

    BASE_URL: str = "https://www.musinsa.com"

    CATEGORIES: dict = {
        "tops": "001",
        "outerwear": "002",
        "bottoms": "003",
        "headwear": "120",
        "activewear": "017",
    }

    PARAMETERS: dict = {
        "gf": "M"
    }

    name = "musinsa_category"

    custom_settings = {
        "USER_AGENT": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/140.0.0.0 Safari/537.36"
        ),
    }

    async def start(self):
        for category in self.CATEGORIES.values():
            url = (
                f"{self.BASE_URL}/category/{category}/goods"
                f"?{urlencode(self.PARAMETERS)}"
            )

            yield scrapy.Request(url)

    def parse(self, response: Response, **kwargs) -> Self:
        products = response.css('a[class^="GoodsItem"]')

        for product in products[:self.COUNTS]:
            product.css('::attr(href)').get()
