# Poe OpenAI 兼容 API 配置说明

## 1. 目的

本说明用于在本项目中快速接入 Poe 的 OpenAI-compatible API，并验证是否可以通过同一套接口访问：

- `Claude`
- `Gemini`
- `GPT`

本地已新增脚本：

- `scripts/poe_openai_compatible_api.py`

该脚本不依赖第三方 `openai` 包，直接走标准 HTTP 请求，因此更容易在当前环境中快速使用。

## 2. 官方文档

Poe 官方文档：

- `https://creator.poe.com/docs/external-applications/openai-compatible-api`

根据官方文档，Poe 提供 OpenAI 兼容 API，因此可以使用类似 OpenAI SDK 或标准 HTTP 的方式访问不同模型。

## 3. 需要准备的内容

需要先在环境变量中配置：

```bash
export POE_API_KEY="你的_poe_api_key"
```

脚本默认 API base URL 为：

```text
https://api.poe.com/v1
```

## 4. 当前脚本支持的功能

### 4.1 列出可用模型

```bash
python scripts/poe_openai_compatible_api.py list-models
```

按关键字过滤：

```bash
python scripts/poe_openai_compatible_api.py list-models --contains Claude
python scripts/poe_openai_compatible_api.py list-models --contains Gemini
python scripts/poe_openai_compatible_api.py list-models --contains GPT
```

如果要看原始 JSON：

```bash
python scripts/poe_openai_compatible_api.py list-models --json
```

### 4.2 使用 `/v1/chat/completions`

脚本内置了 3 个便捷别名：

- `claude` -> `Claude-Sonnet-4.6`
- `gemini` -> `Gemini-3.1-Pro`
- `gpt` -> `GPT-5.4`

示例：

```bash
python scripts/poe_openai_compatible_api.py chat --model claude --prompt "用一句话介绍你自己"
python scripts/poe_openai_compatible_api.py chat --model gemini --prompt "用一句话介绍你自己"
python scripts/poe_openai_compatible_api.py chat --model gpt --prompt "用一句话介绍你自己"
```

也可以直接写完整模型名：

```bash
python scripts/poe_openai_compatible_api.py chat --model Claude-Sonnet-4.6 --prompt "Hello"
```

带 system prompt：

```bash
python scripts/poe_openai_compatible_api.py chat \
  --model claude \
  --system "You are a concise assistant." \
  --prompt "Summarize genomic selection in one sentence."
```

如果需要传 Poe 自定义 bot 参数，可通过 `--extra-body` 透传：

```bash
python scripts/poe_openai_compatible_api.py chat \
  --model claude \
  --prompt "Explain residual fusion briefly." \
  --extra-body '{"reasoning_effort":"high"}'
```

### 4.3 使用 `/v1/responses`

```bash
python scripts/poe_openai_compatible_api.py responses --model gpt --prompt "给我一个简短摘要"
```

带 reasoning effort：

```bash
python scripts/poe_openai_compatible_api.py responses \
  --model gpt \
  --prompt "Compare BayesB and GBLUP briefly." \
  --reasoning-effort high
```

带 web search：

```bash
python scripts/poe_openai_compatible_api.py responses \
  --model gpt \
  --prompt "Search recent genomic selection benchmark trends." \
  --web-search
```

## 5. 是否支持 Claude / Gemini / GPT

从 Poe 官方 OpenAI-compatible API 文档看，答案是：

- 支持

但需要注意两点：

1. 是否能调用成功，取决于你的 Poe 账户 API 权限以及该模型是否对当前账户开放。
2. 最稳妥的方式不是假设模型名，而是先运行：

```bash
python scripts/poe_openai_compatible_api.py list-models
```

然后从返回结果中确认当前可用的实际模型 ID。

也就是说：

- 接口层面支持通过同一套 API 访问 Claude / Gemini / GPT
- 账户层面能否实际调用，要以 `list-models` 返回结果为准

## 6. 推荐验证流程

建议按以下顺序验证：

1. 配置 API key
2. 跑：

```bash
python scripts/poe_openai_compatible_api.py list-models --contains Claude
python scripts/poe_openai_compatible_api.py list-models --contains Gemini
python scripts/poe_openai_compatible_api.py list-models --contains GPT
```

3. 如果都能列出模型，再分别做最小调用测试：

```bash
python scripts/poe_openai_compatible_api.py chat --model claude --prompt "ping"
python scripts/poe_openai_compatible_api.py chat --model gemini --prompt "ping"
python scripts/poe_openai_compatible_api.py chat --model gpt --prompt "ping"
```

4. 如果某个别名失败，就把 `list-models` 返回的真实模型 ID 直接填进 `--model`

## 7. 当前结论

当前我已经在代码层面完成了 Poe OpenAI-compatible API 的接入脚本，支持：

- 列模型
- 调 `/v1/chat/completions`
- 调 `/v1/responses`
- 同一套脚本切换 `Claude / Gemini / GPT`

是否能成功实际访问这三类模型，下一步只需要你提供可用的 `POE_API_KEY`，或者你本地自行执行 `list-models` 验证即可。
