# MUSINSA Dataset Completion v2 — Image Requirements

이 프로젝트는 현재 500개 데이터셋의 **이미지 필수 요구사항**을 채우기 위한 단계입니다.

실제 현재 데이터에서 이미지 manifest를 생성했습니다.

- 상품: **500개**
- 전체 이미지 URL: **7,540개**
- thumbnail: **500**
- goods_gallery: **2,928**
- goods_contents: **4,112**

## 이번 단계의 목표

상품별로 다음을 확보합니다.

1. `front` — 정면 대표 컷
2. `back` — 후면 대표 컷
3. `texture` — 소재/질감 디테일 컷
4. `transparent_cutout_path` — 모델/배경을 제거하고 상품 본체만 남긴 PNG

## 중요한 설계

이미지 역할을 URL 순서만으로 단정하지 않습니다.

- CLIP zero-shot으로 `front / back / texture / product_only / model_worn / other` 점수를 계산
- score threshold 이하인 상품은 **manual review queue**
- garment segmentation 모델로 상품 카테고리 영역만 mask
- 모델 착용 컷에서도 사람 전체가 아니라 의류/신발/모자 클래스만 남기도록 설계

사용 모델 기본값:

```text
image classification : openai/clip-vit-base-patch32
garment segmentation : mattmdjaga/segformer_b2_clothes
```

모델은 `config.json`에서 교체 가능합니다.

## 설치

Python 3.10~3.12 권장.

```bash
python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
```

CUDA가 있으면 자동 사용하고, 없으면 CPU로 작동합니다.

## STEP 1 — 이미지 다운로드

```bash
python -m pipeline.download_images
```

전체 7,540장을 무조건 받지는 않습니다.

상품당:
- thumbnail
- goods_gallery
- goods_contents 일부

를 우선으로 최대 **18장**만 후보로 받아,
500개 기준 최대 약 9,000장 이하로 제한합니다.

실제 데이터의 평균 이미지 수가 약 15장이므로 대부분 원본 후보를 그대로 포함합니다.

## STEP 2 — 역할 분류 + 누끼

```bash
python -m pipeline.classify_and_cutout
```

출력:

```text
outputs/products_dataset_completed_v2.jsonl
outputs/cutouts/{product_id}.png
```

상품에는 아래가 추가됩니다.

```json
{
  "selected_images": {
    "front": {"image_url": "...", "scores": {...}},
    "back": {"image_url": "...", "scores": {...}},
    "texture": {"image_url": "...", "scores": {...}}
  },
  "transparent_cutout_path": "outputs/cutouts/12345.png",

  "has_front_image": true,
  "has_back_image": true,
  "has_texture_image": true,
  "has_transparent_cutout": true,

  "manual_review_required": false
}
```

## STEP 3 — 완성도 검사

```bash
python -m pipeline.audit_v2
```

최종적으로 확인할 핵심 지표:

```text
front / 500
back / 500
texture / 500
cutout / 500
fully image-ready / 500
manual review / 500
```

## 왜 자동 결과를 100% 정답으로 취급하지 않나요?

`front/back/texture`는 사이트가 명시적으로 제공한 메타데이터가 아닙니다.

따라서 v2는:

```text
AI 분류
→ confidence
→ threshold
→ 낮은 confidence만 manual review
```

구조를 사용합니다.

즉 `front`라고 무조건 찍어 넣지 않고,
확신이 부족한 상품은 명시적으로 검수 대상으로 남깁니다.

## 누끼의 의미

일반적인 background removal은 모델의 몸까지 그대로 남길 수 있습니다.

이번 코드는 fashion segmentation을 사용해서 카테고리별로:

```text
top / outer → upper-clothes, coat
bottom      → pants
shoes       → left-shoe, right-shoe
hat         → hat
```

영역만 alpha mask로 남깁니다.

따라서 요구사항인:

> 모델의 포즈/얼굴/배경 노이즈 제거 + 옷 본체 분리

에 더 가까운 처리입니다.

## 아직 건드리지 않는 것

- Fashion-CLIP embedding
- Qdrant
- Elasticsearch
- RRF
- Style / Mood / TPO 생성

이들은 사용자가 요청한 대로 Dataset Completion 이후로 미룹니다.
