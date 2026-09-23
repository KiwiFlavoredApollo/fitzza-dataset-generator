from pathlib import Path
import json, torch, transformers
from transformers import CLIPModel, CLIPProcessor
ROOT=Path(__file__).resolve().parents[1]
cfg=json.loads((ROOT/"config.json").read_text(encoding="utf-8"))
print("torch:",torch.__version__)
print("transformers:",transformers.__version__)
print("cuda:",torch.cuda.is_available())
model=CLIPModel.from_pretrained(cfg["classification_model"])
processor=CLIPProcessor.from_pretrained(cfg["classification_model"])
inputs=processor(text=["a front view of a shirt"],return_tensors="pt",padding=True)
out=model.get_text_features(**inputs)
print("get_text_features type:",type(out))
if torch.is_tensor(out): print("tensor shape:",tuple(out.shape))
elif hasattr(out,"pooler_output") and out.pooler_output is not None: print("pooler_output shape:",tuple(out.pooler_output.shape))
print("SMOKE TEST OK")
