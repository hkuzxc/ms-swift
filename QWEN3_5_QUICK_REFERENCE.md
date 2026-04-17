# Qwen3.5 快速参考表

## 核心代码位置速查

| 功能 | 文件 | 行号 | 说明 |
|------|------|------|------|
| **模型常量** | swift/model/constant.py | 149-150 | 定义 `MLLMModelType.qwen3_5` 和 `qwen3_5_moe` |
| **Qwen3_5MoeLoader** | swift/model/models/qwen.py | 1134-1167 | MoE 版本模型加载器及注册 |
| **Qwen3_5Loader** | swift/model/models/qwen.py | 1170-1203 | 标准版本模型加载器及注册 |
| **Qwen3VLLoader** | swift/model/models/qwen.py | 1059-1071 | 父类加载器，处理 Qwen3 视觉特性 |
| **Qwen3_5Template** | swift/template/templates/qwen.py | 565-581 | 模板定义及注册 |
| **Qwen3_5AgentTemplate** | swift/agent_template/qwen3_coder.py | 166-183 | Agent/工具调用模板 |
| **模型架构** | swift/model/model_arch.py | 30/535-550 | 多模态架构定义 (qwen2_vl) |
| **特殊处理** | swift/utils/transformers_utils.py | 328 | in_proj_a/in_proj_b 参数处理 |
| **分布式支持** | swift/megatron/arguments/megatron_args.py | 657-658 | Q/K LayerNorm 权重衰减支持 |
| **DeepSpeed Z3** | swift/model/register.py | 439-441 | MoE 块配置 |

---

## 模型列表一览

### 标准版本 (Qwen3_5Loader)
```
Qwen/Qwen3.5-0.8B, 2B, 4B, 9B, 27B
Qwen/Qwen3.5-{0.8B,2B,4B,9B}-Base
Qwen/Qwen3.5-27B-FP8
```

### MoE 版本 (Qwen3_5MoeLoader)
```
Qwen/Qwen3.5-{35B-A3B, 122B-A10B, 397B-A17B}[-Base][-FP8]
Qwen/Qwen3.6-{35B-A3B}[-FP8]
```

---

## 代码搜索结果摘要

### Qwen3_5 相关代码出现的文件和行号

| 出现位置 | 关键内容 |
|---------|---------|
| tests/test_align/test_template/test_agent.py:400-402 | Agent 模板测试 |
| swift/utils/transformers_utils.py:328 | 特殊参数处理 |
| swift/template/templates/qwen.py:565-581 | Qwen3_5Template 定义和注册 |
| swift/template/constant.py:143 | TemplateType.qwen3_5 定义 |
| swift/model/register.py:439-441 | DeepSpeed Z3 配置 |
| swift/model/models/qwen.py:1134-1203 | 完整的 Loader 和模型注册 |
| swift/model/constant.py:149-150 | MLLMModelType 定义 |
| swift/megatron/arguments/megatron_args.py:657-658 | Megatron 支持 |
| swift/agent_template/mapping.py:11/27 | Agent 模板映射 |
| swift/agent_template/qwen3_coder.py:166-183 | Qwen3_5AgentTemplate 实现 |

---

## Qwen3_5MoeForConditionalGeneration 出现位置

```
swift/model/models/qwen.py:1137      -> from transformers import ...
swift/model/models/qwen.py:1138      -> self.auto_model_cls = ...
swift/model/models/qwen.py:1165      -> architectures=['Qwen3_5MoeForConditionalGeneration']
```

---

## qwen3_5_moe 出现位置

```
swift/utils/transformers_utils.py:328
swift/model/register.py:439,440
swift/model/models/qwen.py:1144
swift/model/constant.py:150
swift/megatron/arguments/megatron_args.py:657,658
```

---

## 关键概念

### 多模态处理
- **模型类型**: `MLLMModelType` (不是 LLMModelType)
- **架构**: `ModelArch.qwen2_vl` (共用 qwen2_vl 架构)
- **模板类**: `Qwen3_5Template` (继承自 Qwen3VLTemplate)
- **Loader 链**: `ModelLoader` → `Qwen2VLLoader` → `Qwen3VLLoader` → `Qwen3_5Loader/Qwen3_5MoeLoader`
- **视觉处理**: image_token_id=248056, video_token_id=248057

### 特性支持
- ✅ 视觉和视频处理 (tags=['vision', 'video'])
- ✅ 思维链 (is_thinking=True)
- ✅ Agent/工具调用 (agent_template='qwen3_5')
- ✅ 量化 (FP8)
- ✅ 分布式训练 (Megatron/DeepSpeed)
- ✅ MoE 支持 (separate loader for MoE)

---

## 依赖版本要求

| 模型版本 | transformers | qwen_vl_utils | decord |
|---------|-------------|---------------|--------|
| Qwen3.5 标准 | >=5.0.0.dev | >=0.0.14 | required |
| Qwen3.5 MoE | >=5.2.0 | >=0.0.14 | required |

---

## 模板配置参数

```python
QwenTemplateMeta(
    MLLMTemplateType.qwen3_5,
    template_cls=Qwen3_5Template,
    default_system=None,                          # 无默认系统提示
    thinking_prefix='<think>\n',                  # 思维开始标签
    non_thinking_prefix='<think>\n\n</think>\n\n',  # 非思维内容前的标签
    agent_template='qwen3_5',                     # Agent 模板关联
    is_thinking=True                              # 启用思维链
)
```

---

## 快速验证命令

```bash
# 查看 Qwen3.5 的所有提及
grep -r "qwen3_5" /Users/zhangxichen1/代码/ms-swift --include="*.py" | wc -l

# 查看 Qwen3_5MoeForConditionalGeneration 的提及
grep -r "Qwen3_5MoeForConditionalGeneration" /Users/zhangxichen1/代码/ms-swift --include="*.py"

# 查看 qwen3_5_moe 的提及
grep -r "qwen3_5_moe" /Users/zhangxichen1/代码/ms-swift --include="*.py"
```

---

## 结论

**MS-Swift 对 Qwen3.5 的支持是完整的、正确的、深度集成的：**

1. 模型被正确注册为多模态 MLLM (而非纯文本 LLM)
2. 视觉和视频能力通过继承链完整保留
3. 两个独立的 Loader 处理标准版和 MoE 版本
4. 支持所有现代训练技术 (分布式、量化、思维链等)
5. 有明确的测试用例验证支持

**没有发现任何不当处理或缺失的集成。**
