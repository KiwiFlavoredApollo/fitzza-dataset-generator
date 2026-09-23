import re
from pathlib import Path

from rembg import remove

INPUT = Path("output/product_detail")
OUTPUT = Path("output/fashion_clip_images")


class BackgroundRemover:
    def __init__(self, input, output):
        self.input = input
        self.output = output

    def run(self):
        for category in self.input.iterdir():
            for goods_no in category.iterdir():
                for directory in ["thumbnail_images", "goods_images"]:
                    self.remove_background(goods_no / directory)

    def remove_background(self, directory: Path) -> None:
        for input_path in directory.iterdir():
            if not str(input_path).lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            output_path: Path = Path(str.replace(str(input_path), str(self.input), str(self.output)))
            output_path = Path(re.sub(r"\.(?:jpg|jpeg)", ".png", str(output_path), flags=re.IGNORECASE))

            with open(input_path, "rb") as input_file:
                input_bytes = input_file.read()
                output_bytes = remove(input_bytes)

            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "wb") as output_file:
                output_file.write(output_bytes)


if __name__ == "__main__":
    BackgroundRemover(
        input=INPUT,
        output=OUTPUT,
    ).run()
