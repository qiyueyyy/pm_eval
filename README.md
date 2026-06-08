# PM-Eval v0.1

PM-Eval 是一个独立的本地评测看板，用于批量评测文本推荐、客服、搜索、内容推荐等 AI 服务。目标服务只需要通过 HTTP 地址和客户端模式接入，PM-Eval 本身不依赖被测系统代码。

## 功能

- 内置 CSV 测试集，也支持页面上传 CSV 替换。
- 支持测试集 CRUD：在线编辑、新增、删除、保存本地 CSV，并按标签、难度和关键词筛选。
- 支持真实接口模式和 Mock 模式。
- 支持规则评分与可选 LLM-as-Judge。
- 自动识别 Bad Case、归因并生成改进建议。
- 使用 SQLite 保存 `eval_cases`、`eval_runs`、`eval_scores`、`bad_cases`。
- 支持“版本对比”Tab，对比两次评测的核心指标、Bad Case 收敛状态和 case 级变化。
- 支持“评测趋势”Tab，展示成功率、平均分、Bad Case 率随版本变化。
- 支持 Bad Case 类型饼图、根因趋势、严重度分布可视化。
- 支持同时评测两个版本/两个 Prompt，并生成 `compare_report.md`。
- 支持可配置评测模板，可在商品推荐、客服、搜索、内容推荐等场景间切换。
- 支持导出 `results.csv`、`bad_cases.csv`、`report.md`。
- 单版本批量评测支持实时进度条，中途停止后会保留并入库已完成结果。

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
OPENAI_MODEL=qwen3.6-plus

PMEVAL_DB_PATH=./data/pm_eval.sqlite
PMEVAL_TEMPLATE_ID=product_recommendation
PMEVAL_PROMPT_NAME=
PMEVAL_PROMPT_TEXT=
```

如果目标接口不可用，保持 `TARGET_CLIENT_MODE=mock`，仍可跑完整评测流程。

## 启动看板

```bash
streamlit run app.py
```

启动后在侧边栏配置目标名称、接口地址、客户端模式、评测场景模板和是否启用 LLM-as-Judge，然后点击“开始评测”。

侧边栏提供“重新读取 .env”按钮。修改 `.env` 后，如果页面输入框仍显示旧值，点击该按钮会重新读取环境配置并清空相关控件状态。

评测运行时页面会展示实时进度、当前 batch、已完成数量和最近完成 case。点击“暂停评测并查看已完成结果”后，系统会等待当前 case 结束，然后保存已完成部分并刷新结果表和导出文件。暂停后可以点击“继续评测剩余 case”，系统会跳过已完成 case，将剩余结果合并到同一个 Batch ID。

## CSV 字段

测试集 CSV 必须包含以下字段：

```text
case_id,user_query,scenario_type,expected_behavior,constraints_json,difficulty,tags
```

## 测试集管理

“测试集管理”Tab 支持：

- 按 `difficulty` 难度筛选。
- 按 `tags` 标签筛选，标签可用逗号、顿号或中文逗号分隔。
- 按关键词搜索 `case_id`、`user_query`、`expected_behavior` 和 `tags`。
- 在线编辑筛选结果，并将修改合并回完整测试集。
- 新增 case、删除选中 case。
- 保存当前测试集到 `data/`，或下载为 CSV。
- 从 `data/*.csv` 加载本地测试集。

`constraints_json` 可按模板场景放入约束字段，例如商品推荐模板常用：

```json
{
  "budget": 200,
  "category": "粉底液",
  "risk_keywords": ["闷痘", "刺激"]
}
```

## 评测模板

模板文件位于：

```text
pmeval/templates/
```

当前内置：

- `product_recommendation.json`: 商品推荐。
- `customer_service.json`: 客服。
- `search.json`: 搜索。
- `content_recommendation.json`: 内容推荐。

模板可以配置：

- 规则检查项、权重和参数。
- Judge 评分维度。
- Bad Case 阈值、类型枚举和根因枚举。

## 版本对比

每次评测都会写入一个 `batch_id`。完成至少两次评测后，可以在“版本对比”Tab 选择基准版本和对比版本，查看：

- 成功率、Bad Case 率、平均 rule_score、响应时间变化。
- Bad Case 收敛、新增问题、未收敛、稳定通过数量。
- Bad Case 类型和根因变化。
- 单 case 的前后评分、问题类型和根因变化。

## 双版本 / 双 Prompt 评测

在“双版本评测”Tab 中填写 Prompt A 和 Prompt B，系统会使用同一测试集、同一模板连续跑两轮评测。真实接口请求会携带：

```text
prompt_name
prompt
system_prompt
```

评测完成后会自动生成对比报告，并保存为：

```text
exports/compare_report.md
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
