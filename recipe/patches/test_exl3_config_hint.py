import json, os
from cuda_exl3.config import Exl3Config
summary = json.load(open("/models/DeepSeek-V4.1-Flash/config.json"))["quantization_config"]
print("hint before:", Exl3Config._model_path_hint, "summary has storage:", "tensor_storage" in summary)
c = Exl3Config.from_config(dict(summary))
print("OK modules:", len(c.modules), "hint after:", Exl3Config._model_path_hint)
c2 = Exl3Config.from_config(dict(summary)); print("OK second:", len(c2.modules))
Exl3Config._model_path_hint = None; os.environ.pop("CUDA_EXL3_MODEL_PATH")
try:
    Exl3Config.from_config(dict(summary)); print("UNEXPECTED: built without hint")
except ValueError:
    print("negative control raised as expected")
