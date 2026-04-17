# Qwen3.5 完整代码片段集合

## 1. 模型注册常量

**文件**: `/Users/zhangxichen1/代码/ms-swift/swift/model/constant.py`
```python
class MLLMModelType:
    qwen3_5 = 'qwen3_5'
    qwen3_5_moe = 'qwen3_5_moe'
```

**文件**: `/Users/zhangxichen1/代码/ms-swift/swift/template/constant.py`
```python
class MLLMTemplateType:
    qwen3_5 = 'qwen3_5'
```

---

## 2. Qwen3_5MoeLoader 完整实现

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
        model_arch=ModelArch.qwen2_vl,
        architectures=['Qwen3_5MoeForConditionalGeneration'],
        requires=['transformers>=5.2.0', 'qwen_vl_utils>=0.0.14', 'decord'],
        tags=['vision', 'video']))
```

---

## 3. Qwen3_5Loader 完整实现

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
        model_arch=ModelArch.qwen2_vl,
        architectures=['Qwen3_5ForConditionalGeneration'],
        requires=['transformers>=5.0.0.dev', 'qwen_vl_utils>=0.0.14', 'decord'],
        tags=['vision', 'video']))
```

---

## 4. Qwen3VLLoader 父类实现

**文件**: `/Users/zhangxichen1/代码/ms-swift/swift/model/models/qwen.py` (第1059-1071行)

```python
class Qwen3VLLoader(Qwen2VLLoader):

    def _check_qwen_vl_utils(self):
        require_version('qwen_vl_utils>=0.0.14')
        compat_qwen_vl_utils(image_patch_size=16)

    def get_model(self, model_dir: str, config, processor, model_kwargs) -> PreTrainedModel:
        from transformers import Qwen3VLForConditionalGeneration
        self.auto_model_cls = self.auto_model_cls or Qwen3VLForConditionalGeneration
        model = super().get_model(model_dir, config, processor, model_kwargs)
        is_moe = getattr(self, 'is_moe', False)
        _compat_qwen3_vl_mixed_data(model.model, processor, is_moe=is_moe)
        return model
```

---

## 5. Qwen3_5Template 模板实现

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

---

## 6. Qwen3_5AgentTemplate 实现

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

---

## 7. Agent 模板映射

**文件**: `/Users/zhangxichen1/代码/ms-swift/swift/agent_template/mapping.py`

```python
from .qwen3_coder import Qwen3_5AgentTemplate, Qwen3CoderAgentTemplate

agent_template_map = {
    ...
    'qwen3_5': Qwen3_5AgentTemplate,
    ...
}
```

---

## 8. 模型架构配置

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

---

## 9. 特殊参数处理

**文件**: `/Users/zhangxichen1/代码/ms-swift/swift/utils/transformers_utils.py` (第320-339行)

```python
if model_type in {'qwen3_next', 'qwen3_5', 'qwen3_5_moe'}:
    suffix_list += ['in_proj_a', 'in_proj_b']
```

**上下文**:
```python
def get_fixed_lora_modules(model):
    if not hasattr(model, 'model_meta') or not hasattr(model, 'model_info'):
        return
    model_arch = model.model_meta.model_arch
    model_type = model.model_meta.model_type
    prefix_list = []
    suffix_list = []
    if model.model_info.is_moe_model:
        suffix_list += ['mlp.gate', 'mlp.shared_expert_gate']
    if model_type in {'qwen3_next', 'qwen3_5', 'qwen3_5_moe'}:
        suffix_list += ['in_proj_a', 'in_proj_b']
    if model_arch is not None:
        for key in ['vision_tower', 'aligner']:
            value = getattr(model_arch, key, None)
            if value:
                prefix_list += value
    suffix_list.append('lm_head')
    res = []
    for n, m in model.named_modules():
        if 'linear' in m.__class__.__name__.lower() and (any(n.endswith(suffix) for suffix in suffix_list)
                                                         or any(n.startswith(prefix) for prefix in prefix_list)):
```

---

## 10. Megatron 分布式训练支持

**文件**: `/Users/zhangxichen1/代码/ms-swift/swift/megatron/arguments/megatron_args.py` (第657-658行)

```python
if self.apply_wd_to_qk_layernorm and self.model_type not in {'qwen3_next', 'qwen3_5', 'qwen3_5_moe'}:
    raise ValueError('apply_wd_to_qk_layernorm is only supported for qwen3_next, qwen3_5 and qwen3_5_moe')
```

---

## 11. DeepSpeed Z3 配置

**文件**: `/Users/zhangxichen1/代码/ms-swift/swift/model/register.py` (第439-441行)

```python
elif hf_model_type == 'qwen3_5_moe':
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock
    z3_leaf_modules = [Qwen3_5MoeSparseMoeBlock]
```

---

## 12. 测试案例

**文件**: `/Users/zhangxichen1/代码/ms-swift/tests/test_align/test_template/test_agent.py` (第400-402行)

```python
def test_qwen3_5():
    agent_template = agent_template_map['qwen3_5']()
    engine = TransformersEngine('Qwen/Qwen3.5-35B-A3B')
```

---

## 关键导入语句

### 在 Loader 中：
```python
from transformers import Qwen3_5ForConditionalGeneration
from transformers import Qwen3_5MoeForConditionalGeneration
from transformers import Qwen3VLForConditionalGeneration
```

### 在 DeepSpeed 配置中：
```python
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock
```

---

## 模型初始化流程

```
用户请求加载 Qwen/Qwen3.5-27B
    ↓
ModelMeta 注册查询 (qwen3_5 → Qwen3_5Loader)
    ↓
Qwen3_5Loader.get_model()
    ↓
导入 Qwen3_5ForConditionalGeneration
    ↓
调用 Qwen2VLLoader.get_model()
    ↓
Qwen3VLLoader 特殊处理
    ↓
_compat_qwen3_vl_mixed_data() 处理混合数据
    ↓
返回加载的多模态模型
```

---

## 关键特性检查清单

- [x] 模型常量定义 (MLLMModelType.qwen3_5)
- [x] 模板常量定义 (MLLMTemplateType.qwen3_5)
- [x] Loader 实现 (Qwen3_5Loader & Qwen3_5MoeLoader)
- [x] 模板实现 (Qwen3_5Template)
- [x] Agent 模板 (Qwen3_5AgentTemplate)
- [x] 架构配置 (ModelArch.qwen2_vl)
- [x] 特殊参数处理 (in_proj_a/in_proj_b)
- [x] 分布式训练支持 (Megatron)
- [x] MoE 支持 (DeepSpeed Z3)
- [x] 测试用例 (test_qwen3_5)
- [x] 思维链支持 (is_thinking=True)
- [x] 视觉视频支持 (tags=['vision', 'video'])

