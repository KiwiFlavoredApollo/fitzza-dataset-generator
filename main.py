# remove_background.py
import os
from rembg import remove
from PIL import Image

INPUT_DIR = "images"
OUTPUT_DIR = "images_cutout"
os.makedirs(OUTPUT_DIR, exist_ok=True)

for filename in os.listdir(INPUT_DIR):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    input_path = os.path.join(INPUT_DIR, filename)
    output_filename = os.path.splitext(filename)[0] + ".png"  # 투명배경은 png로 저장
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    with open(input_path, "rb") as inp:
        input_bytes = inp.read()
        output_bytes = remove(input_bytes)

    with open(output_path, "wb") as out:
        out.write(output_bytes)

    print(f"완료: {filename} -> {output_filename}")

print(f"\n총 {len(os.listdir(OUTPUT_DIR))}개 누끼 이미지 저장됨: {OUTPUT_DIR}/")