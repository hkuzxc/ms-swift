# MS-Swift Qwen3.5 支持分析报告

## 执行摘要

MS-Swift **完全支持** Qwen3.5 模型架构，包括标准版本和 MoE（Mixture of Experts）版本。这些模型被正确注册为**多模态模型（MLLM）**，具有视觉和视频处理能力。

---

## 1. 模型架构注册情况

### 1.1 模型类型定义

**文件**: `/Users/zhangxichen1/代码/ms-swift/swift/model/constant.py` (第149-150行)

```python
class MLLMModelType:
    qwen3_5 = 'qwen3_5'
    qwen3_5_moe = 'qwen3_5_moe'
```

### 1.2 模板类型定义

**文件**: `/Users/zhangxichen1/代码/ms-swift/swift/template/constant.py` (第143行)

```python
class MLLMTemplateType:
    qwen3_5 = 'qwen3_5'
```

**重要**: 注意模板定义中**只有** `qwen3_5` 一个条目，不包含 `qwen3_5_moe` 的独立模板（共用相同的模板）。

---

## 2. Loader 类实现

### 2.1 Qwen3_5MoeLoader（MoE 版本）

**文件**: `/Users/zhangxichen1/代码/ms-swift/swift/model/models/qwen.py` (第1134-1167行)

```python
class Qwen3_5MoeLoader(Qwen3VLLoader):

    def get_model(self, model_dir: str, config, processor, model_kwargs) -> PreTrainedModel:
        from transformers import Qwen3_5MoeForConditionalGeneration
        self.auto_model_cls = self.auto_model_cls or Qwen3_5MoeForConditionalGeneration
        return Qwen2VLLoader.get_model(self, model_dir, config, processor, model_kwargs)


register_model(
    ModelMeta(
        MLLMModelType.qwen3_5_moe,
        [
            ModelGroup(
                [
                    Model('Qwen/Qwen3.5-35B-A3B-Base', 'Qwen/Qwen3.5-35B-A3B-Base'),
                    Model('Qwen/Qwen3.5-35B-A3B', 'Qwen/Qwen3.5-35B-A3B'),
                    Model('Qwen/Qwen3.5-122B-A10B', 'Qwen/Qwen3.5-122B-A10B'),
                    Model('Qwen/Qwen3.5-397B-A17B', 'Qwen/Qwen3.5-397B-A17B'),
                    # FP8
                    Model('Qwen/Qwen3.5-35B-A3B-FP8', 'Qwen/Qwen3.5-35B-A3B-FP8'),
                    Model('Qwen/Qwen3.5-122B-A10B-FP8', 'Qwen/Qwen3.5-122B-A10B-FP8'),
                    Model('Qwen/Qwen3.5-397B-A17B-FP8', 'Qwen/Qwen3.5-397B-A17B-FP8'),
                ],
                TemplateType.qwen3_5),
            ModelGroup([
                Model('Qwen/Qwen3.6-35B-A3B', 'Qwen/Qwen3.6-35B-A3B'),
                Model('Qwen/Qwen3.6-35B-A3B-FP8', 'Qwen/Qwen3.6-35B-A3B-FP8'),
            ], TemplateType.qwen3_5),
        ],
        Qwen3_5MoeLoader,
        model_arch=ModelArch.qwen2_vl,  # ⚠️ 使用 qwen2_vl 架构
        architectures=['Qwen3_5MoeForConditionalGeneration'],
        requires=['transformers>=5.2.0', 'qwen_vl_utils>=0.0.14', 'decord'],
        tags=['vision', 'video']))
```

**关键特点**:
- 父类: `Qwen3VLLoader`
- 模型类: `Qwen3_5MoeForConditionalGeneration`
- 支持的模型: 35B/122B/397B MoE 版本（含 FP8 量化）和 Qwen3.6 版本
- 架构: `ModelArch.qwen2_vl` （多模态视觉-语言架构）
- 标签: `['vision', 'video']` 表示支持图像和视频

### 2.2 Qwen3_5Loader（标准版本）

**文件**: `/Users/zhangxichen1/代码/ms-swift/swift/model/models/qwen.py` (第1170-1203行)

```python
class Qwen3_5Loader(Qwen3VLLoader):

    def get_model(self, model_dir: str, config, processor, model_kwargs) -> PreTrainedModel:
        from transformers import Qwen3_5ForConditionalGeneration
        self.auto_model_cls = self.auto_model_cls or Qwen3_5ForConditionalGeneration
        return Qwen2VLLoader.get_model(self, model_dir, config, processor, model_kwargs)


register_model(
    ModelMeta(
        MLLMModelType.qwen3_5,
        [
            ModelGroup(
                [
                    Model('Qwen/Qwen3.5-0.8B', 'Qwen/Qwen3.5-0.8B'),
                    Model('Qwen/Qwen3.5-2B', 'Qwen/Qwen3.5-2B'),
                    Model('Qwen/Qwen3.5-4B', 'Qwen/Qwen3.5-4B'),
                    Model('Qwen/Qwen3.5-9B', 'Qwen/Qwen3.5-9B'),
                    Model('Qwen/Qwen3.5-27B', 'Qwen/Qwen3.5-27B'),
                    # FP8
                    Model('Qwen/Qwen3.5-27B-FP8', 'Qwen/Qwen3.5-27B-FP8'),
                    # base
                    Model('Qwen/Qwen3.5-0.8B-Base', 'Qwen/Qwen3.5-0.8B-Base'),
                    Model('Qwen/Qwen3.5-2B-Base', 'Qwen/Qwen3.5-2B-Base'),
                    Model('Qwen/Qwen3.5-4B-Base', 'Qwen/Qwen3.5-4B-Base'),
                    Model('Qwen/Qwen3.5-9B-Base', 'Qwen/Qwen3.5-9B-Base'),
                ],
                TemplateType.qwen3_5),
        ],
        Qwen3_5Loader,
        model_arch=ModelArch.qwen2_vl,  # ⚠️ 使用 qwen2_vl 架构
        architectures=['Qwen3_5ForConditionalGeneration'],
        requires=['transformers>=5.0.0.dev', 'qwen_vl_utils>=0.0.14', 'decord'],
        tags=['vision', 'video']))
```

**关键特点**:
- 父类: `Qwen3VLLoader`
- 模型类: `Qwen3_5ForConditionalGeneration`
- 支持的模型: 0.8B/2B/4B/9B/27B（含 Base 版本和 FP8 量化）
- 架构: `ModelArch.qwen2_vl` （多模态视觉-语言架构）
- 标签: `['vision', 'video']` 表示支持图像和视频

---

## 3. 继承层次结构

```
ModelLoader (基类)
    ↓
Qwen2VLLoader (处理 Qwen2 视觉模型)
    ↓
Qwen3VLLoader (处理 Qwen3 视觉模型，覆盖 get_model 和 _check_qwen_vl_utils)
    ↓
├─ Qwen3_5Loader (处理标准 Qwen3.5，导入 Qwen3_5ForConditionalGeneration)
└─ Qwen3_5MoeLoader (处理 Qwen3.5 MoE，导入 Qwen3_5MoeForConditionalGeneration)
```

### Qwen3VLLoader 实现细节

**文件**: `/Users/zhangxichen1/代码/ms-swift/swift/model/models/qwen.py` (第1059-1071行)

```python
class Qwen3VLLoader(Qwen2VLLoader):

    def _check_qwen_vl_utils(self):
        require_version('qwen_vl_utils>=0.0.14')
        compat_qwen_vl_utils(image_patch_size=16)  # 使用 16 patch size（而非 Qwen2 的 14）

    def get_model(self, model_dir: str, config, processor, model_kwargs) -> PreTrainedModel:
        from transformers import Qwen3VLForConditionalGeneration
        self.auto_model_cls = self.auto_model_cls or Qwen3VLForConditionalGeneration
        model = super().get_model(model_dir, config, processor, model_kwargs)
        is_moe = getattr(self, 'is_moe', False)
        _compat_qwen3_vl_mixed_data(model.model, processor, is_moe=is_moe)  # 处理混合数据兼容性
        return model
```

---

## 4. 模板实现

### 4.1 Qwen3_5Template

**文件**: `/Users/zhangxichen1/代码/ms-swift/swift/template/templates/qwen.py` (第565-581行)

```python
class Qwen3_5Template(Qwen3VLTemplate):
    image_token_id = 248056
    video_token_id = 248057

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return Qwen2VLTemplate._post_encode(self, model, inputs)


register_template(
    QwenTemplateMeta(
        MLLMTemplateType.qwen3_5,
        template_cls=Qwen3_5Template,
        default_system=None,
        thinking_prefix='<think>\n',
        non_thinking_prefix='<think>\n\n</think>\n\n',
        agent_template='qwen3_5',
        is_thinking=True))
```

**关键特点**:
- 继承自 `Qwen3VLTemplate`
- 定义特定的 token IDs 用于图像和视频（248056 和 248057）
- 支持思维链（thinking）功能：`is_thinking=True`
- 使用思维前缀: `<think>\n`
- 使用非思维前缀: `<think>\n\n</think>\n\n`
- 关联 agent 模板：`'qwen3_5'`

---

## 5. Agent 模板实现

### 5.1 Qwen3_5AgentTemplate

**文件**: `/Users/zhangxichen1/代码/ms-swift/swift/agent_template/qwen3_coder.py` (第166-183行)

```python
class Qwen3_5AgentTemplate(Qwen3CoderAgentTemplate):

    def _format_tools(self, tools: List[Union[str, dict]], system: Optional[str] = None, user_message=None) -> str:
        tool_descs = [json.dumps(self.wrap_tool(tool), ensure_ascii=False) for tool in tools]
        tools_prompt = """# Tools

You have access to the following functions:\n\n<tools>
""" + '\n'.join(tool_descs) + f'\n{TOOL_DESC_SUFFIX}'
        if system:
            tools_prompt += f'\n\n{system}'
        return tools_prompt

    def _get_tool_responses(self, tool_messages):
        res_tool = []
        for tool_message in tool_messages:
            tool_content = tool_message['content']
            res_tool.append(f'<tool_response>\n{tool_content}\n</tool_response>\n')
        return ''.join(res_tool)
```

**文件**: `/Users/zhangxichen1/代码/ms-swift/swift/agent_template/mapping.py`

```python
agent_template_map = {
    ...
    'qwen3_5': Qwen3_5AgentTemplate,
    ...
}
```

---

## 6. 架构配置

### 6.1 模型架构定义

**文件**: `/Users/zhangxichen1/代码/ms-swift/swift/model/model_arch.py` (第30行)

```python
class MLLMModelArch:
    qwen2_vl = 'qwen2_vl'
```

**注意**: 没有单独的 `qwen3_5` 架构定义，Qwen3.5 使用 `qwen2_vl` 架构。

### 6.2 架构注册

**文件**: `/Users/zhangxichen1/代码/ms-swift/swift/model/model_arch.py` (第535-550行)

```python
if transformers_ge_4_52:
    register_model_arch(
        MultiModelKeys(
            MLLMModelArch.qwen2_vl,
            language_model=['model.language_model', 'lm_head'],
            aligner='model.visual.merger',
            vision_tower='model.visual',
        ))
else:
    register_model_arch(
        MultiModelKeys(
            MLLMModelArch.qwen2_vl,
            language_model=['model', 'lm_head'],
            aligner='visual.merger',
            vision_tower='visual',
        ))
```

**架构组件**:
- **language_model**: `['model.language_model', 'lm_head']` - 语言模型部分
- **aligner**: `'model.visual.merger'` - 视觉-语言对齐器
- **vision_tower**: `'model.visual'` - 视觉编码器

---

## 7. 多语言模型处理

### 7.1 Qwen3.5 的特殊处理

**文件**: `/Users/zhangxichen1/代码/ms-swift/swift/utils/transformers_utils.py` (第328行)

```python
if model_type in {'qwen3_next', 'qwen3_5', 'qwen3_5_moe'}:
    suffix_list += ['in_proj_a', 'in_proj_b']
```

**含义**: Qwen3.5 系列模型需要特殊处理 `in_proj_a` 和 `in_proj_b` 参数。

### 7.2 Megatron 分布式训练支持

**文件**: `/Users/zhangxichen1/代码/ms-swift/swift/megatron/arguments/megatron_args.py` (第657-658行)

```python
if self.apply_wd_to_qk_layernorm and self.model_type not in {'qwen3_next', 'qwen3_5', 'qwen3_5_moe'}:
    raise ValueError('apply_wd_to_qk_layernorm is only supported for qwen3_next, qwen3_5 and qwen3_5_moe')
```

**含义**: Qwen3.5 系列模型支持在 Q/K LayerNorm 上应用权重衰减。

### 7.3 DeepSpeed Z3 支持

**文件**: `/Users/zhangxichen1/代码/ms-swift/swift/model/register.py` (第439-441行)

```python
elif hf_model_type == 'qwen3_5_moe':
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock
    z3_leaf_modules = [Qwen3_5MoeSparseMoeBlock]
```

**含义**: Qwen3.5 MoE 模型的稀疏 MoE 块被配置为 DeepSpeed ZeRO-3 的叶子模块。

---

## 8. 支持的 Qwen3.5 模型列表

### 8.1 标准版本（Qwen3_5Loader）

| 模型 | 模型 ID |
|------|--------|
| Qwen3.5 0.8B | Qwen/Qwen3.5-0.8B |
| Qwen3.5 0.8B Base | Qwen/Qwen3.5-0.8B-Base |
| Qwen3.5 2B | Qwen/Qwen3.5-2B |
| Qwen3.5 2B Base | Qwen/Qwen3.5-2B-Base |
| Qwen3.5 4B | Qwen/Qwen3.5-4B |
| Qwen3.5 4B Base | Qwen/Qwen3.5-4B-Base |
| Qwen3.5 9B | Qwen/Qwen3.5-9B |
| Qwen3.5 9B Base | Qwen/Qwen3.5-9B-Base |
| Qwen3.5 27B | Qwen/Qwen3.5-27B |
| Qwen3.5 27B FP8 | Qwen/Qwen3.5-27B-FP8 |

### 8.2 MoE 版本（Qwen3_5MoeLoader）

| 模型 | 模型 ID |
|------|--------|
| Qwen3.5 35B-A3B Base | Qwen/Qwen3.5-35B-A3B-Base |
| Qwen3.5 35B-A3B | Qwen/Qwen3.5-35B-A3B |
| Qwen3.5 35B-A3B FP8 | Qwen/Qwen3.5-35B-A3B-FP8 |
| Qwen3.5 122B-A10B | Qwen/Qwen3.5-122B-A10B |
| Qwen3.5 122B-A10B FP8 | Qwen/Qwen3.5-122B-A10B-FP8 |
| Qwen3.5 397B-A17B | Qwen/Qwen3.5-397B-A17B |
| Qwen3.5 397B-A17B FP8 | Qwen/Qwen3.5-397B-A17B-FP8 |
| Qwen3.6 35B-A3B | Qwen/Qwen3.6-35B-A3B |
| Qwen3.6 35B-A3B FP8 | Qwen/Qwen3.6-35B-A3B-FP8 |

---

## 9. 多模态支持确认

### 问题：Qwen3.5 是否被当作多模态模型处理？

**答案：是的，完全作为多模态模型处理。**

证据:

1. **模型类型**: 使用 `MLLMModelType`（多模态 LLM）而非 `LLMModelType`
2. **模型类**: 使用 `Qwen3_5ForConditionalGeneration` 和 `Qwen3_5MoeForConditionalGeneration`（以 "ConditionalGeneration" 命名，表示多模态）
3. **Loader 继承**: 继承自 `Qwen3VLLoader` → `Qwen2VLLoader` → `ModelLoader`（视觉加载链）
4. **处理器**: 使用 `Processor` 而非仅 `Tokenizer`
5. **架构**: 使用 `ModelArch.qwen2_vl`（包含 vision_tower 和 aligner）
6. **标签**: 标签为 `['vision', 'video']`
7. **模板**: 使用 `Qwen3_5Template` 继承自 `Qwen3VLTemplate`（视觉模板）
8. **处理方法**:
   - `_post_encode` 方法处理视觉输入
   - `_compat_qwen3_vl_mixed_data` 处理混合数据（文本+视觉）
   - 配置特定的 token IDs（image_token_id 和 video_token_id）

---

## 10. 测试证据

**文件**: `/Users/zhangxichen1/代码/ms-swift/tests/test_align/test_template/test_agent.py` (第400-402行)

```python
def test_qwen3_5():
    agent_template = agent_template_map['qwen3_5']()
    engine = TransformersEngine('Qwen/Qwen3.5-35B-A3B')
```

存在明确的 Qwen3.5 模型测试，确认支持。

---

## 11. 依赖要求

### 11.1 Qwen3.5 标准版本

```
transformers>=5.0.0.dev
qwen_vl_utils>=0.0.14
decord
```

### 11.2 Qwen3.5 MoE 版本

```
transformers>=5.2.0
qwen_vl_utils>=0.0.14
decord
```

---

## 12. 总结与建议

### 结论

MS-Swift 对 Qwen3.5 系列模型的支持：
- ✅ **完全支持**标准版本（0.8B-27B）和 MoE 版本（35B-A3B-397B-A17B）
- ✅ **正确注册**为多模态 MLLM 模型
- ✅ **支持视觉和视频处理**
- ✅ **支持思维链（CoT）**功能
- ✅ **支持 Agent/工具调用**
- ✅ **支持分布式训练**（Megatron、DeepSpeed）
- ✅ **支持量化**（FP8）
- ✅ **支持 Base 版本**（未微调的基础模型）

### 注意事项

1. Qwen3.5 模型共用 `Qwen3_5Template` 模板，无论是标准版还是 MoE 版
2. 两个版本都使用 `ModelArch.qwen2_vl` 架构，而不是独立的 `qwen3_5` 架构
3. 需要 `transformers>=5.0.0.dev` 或更高版本
4. 需要 `qwen_vl_utils>=0.0.14` 用于视觉处理
5. 需要 `decord` 用于视频处理

