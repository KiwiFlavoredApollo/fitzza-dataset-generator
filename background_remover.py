import re
from pathlib import Path

import rembg

INPUT = Path("output/product_detail")
OUTPUT = Path("output/fashion_clip_images")


class BackgroundRemover:
    def __init__(self, input, output):
        self.input = input
        self.output = output

    def run(self):
        for category in self.input.iterdir():
            for goods_no in category.iterdir():
                for image in (goods_no / "thumbnail_images").iterdir():
                    self.remove_background(image)

                for image in (goods_no / "goods_images").iterdir():
                    self.remove_background(image)

    def remove_background(self, input: Path) -> None:
        if not str(input).lower().endswith((".jpg", ".jpeg", ".png")):
            return

        output: Path = Path(str.replace(str(input), str(self.input), str(self.output)))
        output = Path(re.sub(r"\.(?:jpg|jpeg)", ".png", str(output), flags=re.IGNORECASE))

        session = rembg.new_session(
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )

        with open(input, "rb") as input_file:
            input_bytes = input_file.read()
            output_bytes = rembg.remove(input_bytes, session=session)

        output.parent.mkdir(parents=True, exist_ok=True)

        with open(output, "wb") as output_file:
            output_file.write(output_bytes)

        print(f"작업완료: {output}")


if __name__ == "__main__":
    BackgroundRemover(
        input=INPUT,
        output=OUTPUT,
    ).run()
