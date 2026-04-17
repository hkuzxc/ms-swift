#!/usr/bin/env python3
"""debug_qwen35_loading.py - 定位 Qwen3.5 MISSING 权重问题

用法:
    conda activate swift_qwen_35
    cd /mnt/tidalfs-bdsz01/usr/xiangyi3/zxc/ms-swift
    python debug_qwen35_loading.py 2>&1 | tee debug_output.txt
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import transformers
print(f"transformers version: {transformers.__version__}")
print(f"torch version: {torch.__version__}")

MODEL_DIR = "/mnt/tidalfs-bdsz01/dataset/llm_ckpt/qwen3.5/Qwen3.5-35B-A3B"

# ====== 测试1: 直接用 transformers 加载 ======
print("\n" + "=" * 60)
print("TEST 1: 直接 transformers from_pretrained")
print("=" * 60)
from transformers import Qwen3_5MoeForConditionalGeneration, AutoConfig

config1 = AutoConfig.from_pretrained(MODEL_DIR, trust_remote_code=True)
print(f"Config type: {type(config1).__name__}")
print(f"Config model_type: {config1.model_type}")
print(f"Has text_config: {hasattr(config1, 'text_config')}")
print(f"Has vision_config: {hasattr(config1, 'vision_config')}")

try:
    model1 = Qwen3_5MoeForConditionalGeneration.from_pretrained(
        MODEL_DIR, config=config1, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map='auto'
    )
    print("TEST 1 SUCCESS: transformers 直接加载成功!")
    total_params = sum(p.numel() for p in model1.parameters())
    print(f"Total parameters: {total_params:,}")
    del model1
    torch.cuda.empty_cache()
except Exception as e:
    print(f"TEST 1 FAILED: {e}")
    import traceback
    traceback.print_exc()

# ====== 测试2: 用 ms-swift get_model_processor 加载 ======
print("\n" + "=" * 60)
print("TEST 2: ms-swift get_model_processor")
print("=" * 60)
try:
    from swift.model.register import get_model_processor
    model2, processor2 = get_model_processor(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
        load_model=True,
    )
    print("TEST 2 SUCCESS: ms-swift 加载成功!")
    print(f"Model type: {type(model2).__name__}")
    total_params = sum(p.numel() for p in model2.parameters())
    print(f"Total parameters: {total_params:,}")
    del model2, processor2
    torch.cuda.empty_cache()
except Exception as e:
    print(f"TEST 2 FAILED: {e}")
    import traceback
    traceback.print_exc()

# ====== 测试3: 逐步模拟 ms-swift 加载流程 ======
print("\n" + "=" * 60)
print("TEST 3: 逐步模拟 ms-swift 加载流程")
print("=" * 60)

from swift.model.register import MODEL_MAPPING

model_meta = MODEL_MAPPING.get('qwen3_5_moe')
print(f"Model meta found: {model_meta is not None}")
if model_meta:
    print(f"  model_arch: {model_meta.model_arch}")
    print(f"  architectures: {model_meta.architectures}")
    print(f"  is_multimodal: {model_meta.is_multimodal}")

# 测试 config postprocess 是否改了什么关键的东西
config3 = AutoConfig.from_pretrained(MODEL_DIR, trust_remote_code=True)
from swift.utils.hf_config import HfConfigFactory

print(f"\nConfig before postprocess:")
print(f"  type: {type(config3).__name__}")
print(f"  hidden_size: {getattr(config3, 'hidden_size', 'N/A')}")
print(f"  text_config.hidden_size: {getattr(config3.text_config, 'hidden_size', 'N/A') if hasattr(config3, 'text_config') else 'N/A'}")

HfConfigFactory.set_config_attr(config3, 'torch_dtype', torch.bfloat16, include_vit=True)
HfConfigFactory.compat_zero3(config3)

print(f"\nConfig after postprocess:")
print(f"  type: {type(config3).__name__}")
print(f"  hidden_size: {getattr(config3, 'hidden_size', 'N/A')}")
print(f"  text_config.hidden_size: {getattr(config3.text_config, 'hidden_size', 'N/A') if hasattr(config3, 'text_config') else 'N/A'}")

# 测试用 postprocessed config 调 from_pretrained 是否出问题
print("\n" + "=" * 60)
print("TEST 4: 用 postprocessed config 直接 from_pretrained")
print("=" * 60)
try:
    model4 = Qwen3_5MoeForConditionalGeneration.from_pretrained(
        MODEL_DIR, config=config3, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map='auto'
    )
    print("TEST 4 SUCCESS: postprocessed config 加载成功!")
    total_params = sum(p.numel() for p in model4.parameters())
    print(f"Total parameters: {total_params:,}")
    del model4
    torch.cuda.empty_cache()
except Exception as e:
    print(f"TEST 4 FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("ALL TESTS DONE - 请把完整输出发给我")
print("=" * 60)
