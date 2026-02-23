# nano-vllm 初始搭建分析（从第一个 commit 开始）

## 分析方法（命令）
- `git log --reverse --oneline --decorate --date=short --pretty=format:'%h %ad %an %s'`
- `git show --stat --name-status --format=fuller a5a4909`
- `git show a5a4909:<path>`（逐个查看首个提交关键文件）
- `git show --unified=3 <commit> -- <path...>`（查看早期修复/重构的关键差异）

## 阶段 0：仓库“冷启动”——a5a4909 (init commit)
首个提交一次性加入了完整最小可运行骨架：

1. **入口与对外 API**
   - `nanovllm/__init__.py` 暴露 `LLM` 与 `SamplingParams`。
   - `nanovllm/llm.py` 让 `LLM` 直接继承 `LLMEngine`，说明最开始就是“薄封装 + 核心引擎”的结构。

2. **配置与参数面**
   - `nanovllm/config.py` 用 dataclass 定义核心运行参数：批处理 token 上限、并发序列上限、显存利用率、KV cache block 大小、是否强制 eager 等。
   - `nanovllm/sampling_params.py` 单独抽象采样参数：温度、最大生成长度、是否忽略 eos。

3. **执行主链路（Engine）**
   - `nanovllm/engine/llm_engine.py` 初始化时串起：
     - HF `AutoConfig` / `AutoTokenizer`
     - `ModelRunner`（模型执行）
     - `Scheduler`（调度）
   - `generate()` 主循环是典型的 vLLM 风格：`add_request -> schedule/step -> postprocess -> decode`。

4. **调度与生命周期管理**
   - `nanovllm/engine/sequence.py` 定义请求状态机与 token 缓冲（WAITING/RUNNING/FINISHED）。
   - `nanovllm/engine/scheduler.py` 实现 prefill / decode 两阶段调度。
   - `nanovllm/engine/block_manager.py` 实现分页 KV cache 的 block 分配、释放与哈希复用（prefix cache）。

5. **模型执行与缓存布局**
   - `nanovllm/engine/model_runner.py` 完成：
     - CUDA 设备与 dtype 上下文切换
     - Qwen3 模型实例化
     - KV cache 预分配和挂载到注意力层
     - prefill/decode 输入张量构造
     - （非 eager）CUDA Graph capture

6. **模型与算子层实现**
   - `nanovllm/models/qwen3.py` 内置 Qwen3 CausalLM 结构。
   - `nanovllm/layers/*` 提供 attention、rotary、parallel linear、norm、sampler、embedding/head 等基础组件。

7. **可运行脚本与依赖**
   - `example.py` 提供聊天模板推理例子。
   - `bench.py` 提供吞吐 benchmark。
   - `requirements.txt` 提供最小依赖集（torch/triton/transformers/cmake/ninja）。

> 结论：第一个提交不是“空仓初始化”，而是直接落地了一个端到端可跑通的单模型推理内核（含调度、缓存、模型、示例与 benchmark）。

## 阶段 1：首日修复——b98e1ca (fix)
在初版骨架后，作者马上修了几类“可用性与正确性”问题：

1. **吞吐可观测性增强**
   - `LLMEngine.generate()` 增加 prefill / decode 实时 tok/s 展示。

2. **decode 索引正确性修复**
   - `ModelRunner.prepare_decode()` 的 `slot_mapping` 从 `... + len(last_block())` 改为 `... + len(last_block()) - 1`，避免 off-by-one。

3. **停止条件修复**
   - `Scheduler.postprocess()` 改为尊重 `ignore_eos`（此前会无条件按 eos 停止）。

4. **命名与小问题修正**
   - `capture_model` 重命名为 `capture_cudagraph`。
   - prompt 处理与参数默认值也做了微调（如 batched token 上限加大）。

## 阶段 2：结构重构——386290d (refactor)
这一提交显示作者开始从“先跑起来”转向“更通用和更稳”：

1. **block_size 从硬编码 256 走向参数化**
   - `Config` 约束从“必须等于 256”放宽到“256 的倍数”。
   - `BlockManager/Sequence` 的若干逻辑去掉硬编码，改为跟随配置。

2. **调度策略防御增强**
   - `can_append(seq)` 按序列是否会跨块来判断是否需要新 block，而不是固定判定。
   - 新增 `can_prefill` 逻辑，开始考虑 cache 水位对 prefill 的影响。

3. **输出结构调整**
   - `generate()` 输出由纯文本变为 `{text, token_ids}`，便于上层系统做可观测与后处理。

## 阶段 3：权重加载工程化——08c84ec (multi file loader)
初版 `Qwen3ForCausalLM.load_weights()` 被抽离成通用 `utils/loader.py`：

1. **支持多 safetensors 文件扫描加载**
   - 从“固定读一个 `model.safetensors`”改成遍历目录中的 `*.safetensors`。

2. **packed 权重映射更清晰**
   - 将 `packed_modules_mapping` 从“模块->列表”转成“原始权重名->(目标模块, shard_id)”映射，减少分支判断。

3. **Runner 与模型解耦**
   - `ModelRunner` 直接调用 `load_model(self.model, path)`，后续新增模型时可复用加载器。

## 阶段 4：继续打磨（紧随其后）
- `fee58d4`、`f16adb7` 等提交继续围绕调度/配置/引擎做修复和简化，说明最初 2~3 天主要是在“稳定核心执行链路”。

## 总体判断：这个项目最开始是如何“搭起来”的
1. **先把最短可运行链路一次性铺齐**：API、配置、调度、KV cache、模型定义、加载、示例、基准全在首个提交就到位。
2. **再快速修 correctness 与可观测**：off-by-one、eos 逻辑、吞吐指标第一时间补齐。
3. **接着做工程化抽象**：参数化 block 大小、统一 loader、输出结构化，为后续多模型/多功能演进打基础。
4. **开发节奏符合“高强度原型迭代”**：先全链路可跑，再连发修复与重构，逐步从 Demo 内核走向可维护项目。

---

## 以首个 commit 为标准的“复刻第一步”实现路径

如果你要复刻作者第一步（即 `a5a4909` 的效果），建议按下面路径推进：**先跑通，再优化**。

### 0) 目标定义（先写在 README 顶部）
- 单机单卡。
- 单模型（先固定 Qwen3）。
- 支持批量 prompt 输入。
- 支持 prefill + decode 两阶段调度。
- 支持分页 KV cache（block 化，先固定 256）。
- 提供 `example.py`（功能验证）与 `bench.py`（性能基线）。

> 验收口径：`example.py` 能输出文本，`bench.py` 能输出 tok/s。

### 1) 第 1 天：最小 API + 配置面
1. 建立 `LLM` 与 `SamplingParams` 对外接口。
2. 配置项最小集：
   - `model`
   - `max_num_batched_tokens`
   - `max_num_seqs`
   - `max_model_len`
   - `gpu_memory_utilization`
   - `enforce_eager`
   - `kvcache_block_size`
3. 保持“薄入口”：`LLM` 只做包装，核心逻辑在 `LLMEngine`。

### 2) 第 2 天：先把主执行环跑起来
1. `LLMEngine.__init__` 串起 tokenizer/config/model_runner/scheduler。
2. 先实现同步 `generate()` 循环：
   - `add_request`
   - `schedule`
   - `model_runner.run`
   - `postprocess`
3. 先不做流式，先确保批处理闭环完整。

### 3) 第 3 天：实现调度与序列状态机
1. `Sequence`：保存 prompt token、完成 token、状态。
2. `Scheduler`：
   - waiting/running 两个队列
   - prefill 优先
   - decode 逐步推进
3. `postprocess` 支持终止条件：eos 或 `max_tokens`。

### 4) 第 4 天：实现分页 KV cache（核心）
1. `BlockManager`：
   - free/used block 管理
   - allocate/deallocate
   - append 时按需扩块
2. 加 prefix hash 复用逻辑（先简单可用，再做严格一致性校验）。
3. 先把 block size 固定为 256，减少变量。

### 5) 第 5~6 天：模型执行层与算子拼装
1. `ModelRunner`：
   - 权重加载
   - KV cache 张量预分配
   - prefill/decode 输入准备
2. `models/qwen3.py`：
   - embedding
   - attention + rotary
   - mlp
   - lm_head
3. `layers/*`：只实现当前模型会用到的最小集合。

### 6) 第 7 天：脚本与可用性验证
1. `example.py`：2~3 条 prompt 跑通。
2. `bench.py`：固定 batch/seq/max_tokens 输出吞吐。
3. 记录已知限制（仅单卡、仅某模型、仅离线权重目录）。

---

## 复刻版建议目录（对应首 commit 的结构）

```text
your-nano-vllm/
├── README.md
├── LICENSE
├── requirements.txt
├── example.py
├── bench.py
└── nanovllm/
    ├── __init__.py
    ├── llm.py
    ├── config.py
    ├── sampling_params.py
    ├── engine/
    │   ├── sequence.py
    │   ├── scheduler.py
    │   ├── block_manager.py
    │   ├── model_runner.py
    │   └── llm_engine.py
    ├── models/
    │   └── qwen3.py
    ├── layers/
    │   ├── activation.py
    │   ├── attention.py
    │   ├── embed_head.py
    │   ├── layernorm.py
    │   ├── linear.py
    │   ├── rotary_embedding.py
    │   └── sampler.py
    └── utils/
        ├── context.py
        └── memory.py
```

---

## 可直接落 Jira/GitHub 的 Ticket 拆分（按首步复刻）

下面给你一个可以直接建 issue 的版本（建议按 P0 → P1）：

### P0（必须完成，形成首个可运行版本）

1. **TICKET-001: 初始化项目骨架与依赖**
   - 产物：目录结构 + `requirements.txt` + 空 README。
   - 验收：`python -c "import nanovllm"` 通过。

2. **TICKET-002: 实现公开 API（LLM/SamplingParams）**
   - 产物：`nanovllm/__init__.py`、`llm.py`、`sampling_params.py`。
   - 验收：能构造 `SamplingParams`，`LLM` 能初始化。

3. **TICKET-003: 实现 Config 与 HF 配置接入**
   - 产物：`config.py` + `AutoConfig/AutoTokenizer` 接入。
   - 验收：从本地模型目录读取 config/tokenizer 成功。

4. **TICKET-004: 实现 Sequence 状态机**
   - 产物：`Sequence` + `SequenceStatus`。
   - 验收：token append 与 completion 计数正确。

5. **TICKET-005: 实现 BlockManager（分配/释放）**
   - 产物：`block_manager.py`（先不做复杂优化）。
   - 验收：构造多序列时 block 分配数量正确，无泄漏。

6. **TICKET-006: 实现 Scheduler（prefill/decode）**
   - 产物：`scheduler.py`。
   - 验收：waiting/running 转移正确，满足 eos/max_tokens 停止。

7. **TICKET-007: 实现 Qwen3 最小推理图**
   - 产物：`models/qwen3.py` + `layers/*`。
   - 验收：forward 维度正确，能输出 logits。

8. **TICKET-008: 实现 ModelRunner（KV cache + run）**
   - 产物：`model_runner.py`。
   - 验收：prefill/decode 各跑一步无报错。

9. **TICKET-009: 串联 LLMEngine.generate 主循环**
   - 产物：`llm_engine.py`。
   - 验收：输入多 prompt，返回文本结果。

10. **TICKET-010: 提供 example 与 bench**
    - 产物：`example.py`、`bench.py`。
    - 验收：example 出结果、bench 出吞吐数值。

### P1（紧随首版的稳定性修复）

11. **TICKET-011: 修复 decode slot_mapping off-by-one**
    - 背景：对应早期 `b98e1ca` 的关键修复点。
    - 验收：decode 阶段索引一致，长序列无错位。

12. **TICKET-012: 停止条件支持 ignore_eos**
    - 背景：避免强制 eos 截断。
    - 验收：`ignore_eos=True` 时只由 `max_tokens` 控制。

13. **TICKET-013: 生成过程吞吐可观测（prefill/decode）**
    - 背景：快速定位性能瓶颈。
    - 验收：进度条展示两段 tok/s。

14. **TICKET-014: block_size 参数化（从硬编码走向配置）**
    - 背景：对应早期重构方向。
    - 验收：256/512 等配置可运行。

15. **TICKET-015: 抽离通用权重加载器（多 safetensors）**
    - 背景：对应 multi-file loader 演进。
    - 验收：目录内多分片权重可全部加载。

---

## 你可以直接执行的“第一周节奏”
- 周一~周二：TICKET-001~004
- 周三：TICKET-005~006
- 周四~周五：TICKET-007~009
- 周六：TICKET-010 + 联调
- 周日：TICKET-011~013（最小稳定性补丁）

如果你希望，我下一步可以把这套 tickets 再转成：
- GitHub Projects 的看板格式（Todo / In Progress / Done）
- 每个 ticket 的「预估工时 + 风险 + 回滚方案」模板。
