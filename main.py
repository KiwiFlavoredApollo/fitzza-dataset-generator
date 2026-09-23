from pathlib import Path

from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

from musinsa_crawler.musinsa.categories import *
from musinsa_crawler.spiders.musinsa_product_detail_spider import MusinsaProductDetailSpider
from musinsa_crawler.spiders.musinsa_product_list_spider import MusinsaProductListSpider

CATEGORIES = [
    TOPS,
    OUTERWEAR,
    BOTTOMS,
    HEADWEAR,
    ACTIVEWEAR
]

INPUT = Path("input")
OUTPUT = Path("output")


if __name__ == "__main__":
    process = CrawlerProcess(get_project_settings())

    for category in CATEGORIES:
        process.crawl(
            MusinsaProductDetailSpider,
            category=category,
            input=INPUT,
            output=OUTPUT
        )

    process.start()
