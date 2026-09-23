# v2.2 patch

오류 `BaseModelOutputWithPooling has no attribute norm` 대응 버전입니다.

현재 transformers 버전에서는 `CLIPModel.get_text_features()` / `get_image_features()`가 Tensor 대신 ModelOutput 객체를 반환할 수 있습니다. v2.2는 두 형태를 모두 처리합니다.

먼저:

```bash
python -m pipeline.smoke_test
```

마지막에 `SMOKE TEST OK`가 나오면:

```bash
python -m pipeline.classify_and_cutout
```

HF_TOKEN / symlink 경고는 현재 치명적 오류가 아닙니다.
