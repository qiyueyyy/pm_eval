# PM-Eval v0.1

PM-Eval 是一个独立的本地评测看板 MVP，用于批量评测任意文本推荐或聊天接口。它可以评测 BeautyAgent，也可以切换到其他目标服务；目标服务只通过 HTTP 地址和客户端模式配置接入，PM-Eval 本身不依赖 BeautyAgent 后端代码。

## 功能

- 内置 30 条文本推荐测试用例。
- 支持页面上传 CSV 替换测试集。
- 支持配置目标接口地址、超时和目标名称。
- 支持真实接口模式和 Mock 模式。
- 支持规则评分与可选 LLM-as-Judge。
- 自动识别 Bad Case、归因并生成改进建议。
- 使用 SQLite 保存 eval_cases、eval_runs、eval_scores、bad_cases。
- 支持导出 results.csv、bad_cases.csv、report.md。

## 安装

```bash
cd pm_eval
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 配置

复制环境变量示例：

```bash
copy .env.example .env
```

常用配置：

```env
TARGET_NAME=BeautyAgent
TARGET_API_URL=http://localhost:8000/api/agent/chat
TARGET_API_TIMEOUT=90
TARGET_CLIENT_MODE=mock
TARGET_MOCK_MODE=true

OPENAI_API_KEY=
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini

PMEVAL_DB_PATH=./data/pm_eval.sqlite
```

如果目标接口不可用，保持 `TARGET_MOCK_MODE=true`，仍可跑完整评测流程。

旧版 `BEAUTYAGENT_*` 环境变量仍兼容，但新配置请优先使用 `TARGET_*`。

## 启动看板

```bash
streamlit run app.py
```

启动后在侧边栏配置目标名称、接口地址、是否启用 Mock 模式、是否启用 LLM-as-Judge，然后点击“开始评测”。

## CSV 字段

测试集 CSV 必须包含以下字段：

```text
case_id,user_query,scenario_type,expected_behavior,constraints_json,difficulty,tags
```

`constraints_json` 支持的常用字段：

```json
{
  "budget": 200,
  "categories": ["防晒", "面霜"],
  "risk_keywords": ["闷痘", "刺激"]
}
```

## 输出文件

运行评测后，页面可直接下载，也会自动保存到：

```text
exports/results.csv
exports/bad_cases.csv
exports/report.md
```

SQLite 数据库默认保存到：

```text
data/pm_eval.sqlite
```

## 接口适配

`pmeval/target_client.py` 提供两个通用客户端：

- `MockTargetClient`: 本地 Mock，接口不可用时也能跑完整流程。
- `RealTargetClient`: 调用真实目标接口。

当 `TARGET_API_URL` 为 `/api/agent/chat` 时，客户端会使用 BeautyAgent 兼容表单请求：

```text
message=用户问题
user_id=pm_eval
use_ai=true
```

其他 URL 默认优先发送 POST JSON：

```json
{
  "query": "用户问题",
  "user_query": "用户问题",
  "message": "用户问题",
  "case_id": "CASE_001",
  "constraints": {}
}
```

如果目标接口返回 400、415 或 422，客户端会自动降级尝试 Form 请求。

响应会统一转换为：

```json
{
  "answer": "...",
  "raw_response": {},
  "retrieved_items": [],
  "tool_calls": []
}
```

返回字段会优先读取 `answer`、`assistant_message`、`recommendation`、`response`、`text`、`ai_summary`。真实接口失败只会记录在单条 case 的 `error` 字段，不会中断整批评测。
