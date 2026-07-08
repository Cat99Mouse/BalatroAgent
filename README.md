<div align="center">
  <h1>BalatroAgent</h1>
  <p><em>LLM-driven Balatro agent built on BalatroBot and BalatroLLM.</em></p>
  <div><img src="./docs/assets/balatrollm.svg" alt="BalatroLLM logo" width="150" height="150"></div>
</div>

---

Languages: [English](#english) | [中文](#%E4%B8%AD%E6%96%87)

## English

BalatroAgent is a local automation stack for playing
[Balatro](https://www.playbalatro.com/) with an LLM agent. The executable Python
package is still named `balatrollm`: it starts Balatro through BalatroBot,
renders the current game state into strategy prompts, asks an OpenAI-compatible
model for tool calls, executes those actions in game, and records every run for
debugging and review.

### What It Does

- Runs an LLM-controlled Balatro bot from the `balatrollm` CLI.
- Starts and manages one or more Balatro instances through `balatrobot`.
- Supports OpenAI-compatible providers such as OpenRouter, OpenAI, local
    inference gateways, or any endpoint that supports chat completions with tool
    calling.
- Generates task batches from model, seed, deck, stake, and strategy
    combinations.
- Uses strategy folders made of Jinja prompt templates and JSON tool
    definitions.
- Records run artifacts under `runs/`, including requests, responses, game
    states, screenshots, logs, and final statistics.
- Provides optional local views for inspecting the latest run.

### Requirements

- Balatro installed locally.
- BalatroBot installed and working. `balatrollm` uses the BalatroBot API to
    start instances and execute game actions.
- `uv` for Python environment and dependency management.
- An LLM model that supports tool/function calling.
- Python 3.13 or newer. `uv sync` can install the required Python version
    automatically.

### Quick Start

```powershell
git clone git@github.com:Cat99Mouse/BalatroAgent.git
cd BalatroAgent
uv sync
```

Set your LLM provider credentials:

```powershell
$env:BALATROLLM_BASE_URL = "https://openrouter.ai/api/v1"
$env:BALATROLLM_API_KEY = "sk-..."
```

Preview the task queue without launching Balatro:

```powershell
uv run balatrollm config\example.yaml --dry-run
```

Run the agent:

```powershell
uv run balatrollm config\example.yaml
```

Enable the local run views:

```powershell
uv run balatrollm config\example.yaml --views
```

Then open:

```text
http://localhost:12345/views/task.html
http://localhost:12345/views/responses.html
```

### Configuration

Configuration is loaded in this order, from lowest to highest priority:

1. `BALATROLLM_*` environment variables
2. YAML config file
3. CLI arguments

The config file can be passed directly:

```powershell
uv run balatrollm config\example.yaml
```

Or through an environment variable:

```powershell
$env:BALATROLLM_CONFIG = "config\example.yaml"
uv run balatrollm
```

Common options:

| Option       | Environment Variable  | Default     | Purpose                      |
| ------------ | --------------------- | ----------- | ---------------------------- |
| `--model`    | `BALATROLLM_MODEL`    | required    | LLM model name               |
| `--seed`     | `BALATROLLM_SEED`     | `AAAAAAA`   | Balatro seed                 |
| `--deck`     | `BALATROLLM_DECK`     | `RED`       | Deck code                    |
| `--stake`    | `BALATROLLM_STAKE`    | `WHITE`     | Stake code                   |
| `--strategy` | `BALATROLLM_STRATEGY` | `default`   | Strategy folder              |
| `--parallel` | `BALATROLLM_PARALLEL` | `1`         | Concurrent Balatro instances |
| `--host`     | `BALATROLLM_HOST`     | `127.0.0.1` | BalatroBot host              |
| `--port`     | `BALATROLLM_PORT`     | `12346`     | First BalatroBot port        |
| `--base-url` | `BALATROLLM_BASE_URL` | OpenRouter  | LLM API base URL             |
| `--api-key`  | `BALATROLLM_API_KEY`  | unset       | LLM API key                  |
| `--views`    | `BALATROLLM_VIEWS`    | `0`         | Start local views server     |

Multiple values are supported for `model`, `seed`, `deck`, `stake`, and
`strategy`. The agent creates the cartesian product of those values:

```powershell
uv run balatrollm --model openai/gpt-4o --deck RED BLUE --seed AAAAAAA BBBBBBB --dry-run
```

### Strategies

Strategies live in `src/balatrollm/strategies/`. Built-in strategies include:

- `default`
- `aggressive`
- `conservative`

Each strategy directory must contain:

```text
STRATEGY.md.jinja
GAMESTATE.md.jinja
MEMORY.md.jinja
TOOLS.json
```

`STRATEGY.md.jinja` describes long-term behavior, `GAMESTATE.md.jinja` renders
the current state, `MEMORY.md.jinja` keeps recent action and error context, and
`TOOLS.json` defines valid tool calls for each game state. During hand
selection, the bot can also expose a read-only `score_candidates` observation
tool so the model can compare candidate plays before committing to an action.

### Run Artifacts

Every run is written under `runs/`:

```text
runs/
  latest.json
  v<version>/<strategy>/<vendor>/<model>/<timestamp>_<deck>_<stake>_<seed>/
    task.json
    strategy.json
    run.log
    requests.jsonl
    responses.jsonl
    gamestates.jsonl
    stats.json
    screenshots/
```

These files are useful for replaying prompts, checking model behavior,
debugging failed tool calls, and comparing strategies or models.

### Project Layout

```text
config/                  Example YAML configs
docs/                    MkDocs documentation
scripts/                 Local helper and test scripts
src/balatrollm/          Python package and CLI implementation
src/balatrollm/strategies/
tests/                   Unit tests
views/                   Local HTML views for latest run artifacts
```

If you later vendor local BalatroBot mod code into this repository, keep it as a
separate top-level folder such as `balatrobot/` or `mods/balatrobot/` instead of
mixing it into the Python package.

### Development

```powershell
uv sync
uv run pytest
uv run balatrollm --help
```

Local secrets and generated data should not be committed. Keep API keys in
environment variables or a local ignored file, and keep `runs/`, `logs/`,
`.venv/`, and local config files out of version control.

## 中文

BalatroAgent 是一套本地运行的 Balatro 自动游玩系统。实际的 Python 包和命令名仍然是
`balatrollm`：它通过 BalatroBot 启动和控制游戏实例，把当前游戏状态渲染成策略
prompt，调用兼容 OpenAI Chat Completions 的模型，让模型返回工具调用，然后把动作
执行到游戏里，并保存完整运行记录用于调试和复盘。

### 功能概览

- 通过 `balatrollm` CLI 运行 LLM 控制的 Balatro bot。
- 通过 `balatrobot` 启动、连接和管理一个或多个 Balatro 实例。
- 支持 OpenRouter、OpenAI、本地推理网关等兼容 OpenAI 接口且支持工具调用的模型。
- 支持按模型、种子、牌组、难度、策略生成批量任务。
- 策略由 Jinja prompt 模板和 JSON 工具定义组成，便于单独迭代打法。
- 每局运行都会写入 `runs/`，包括请求、响应、游戏状态、截图、日志和统计数据。
- 可选启动本地 views 页面查看最新运行结果。

### 环境要求

- 本地已安装 Balatro。
- BalatroBot 已安装并可正常工作。`balatrollm` 依赖 BalatroBot API 来启动游戏和执行动作。
- 已安装 `uv`，用于管理 Python 环境和依赖。
- LLM 模型必须支持工具调用，也就是 function calling。
- Python 3.13 或更高版本。通常执行 `uv sync` 时会自动准备所需 Python 版本。

### 快速开始

```powershell
git clone git@github.com:Cat99Mouse/BalatroAgent.git
cd BalatroAgent
uv sync
```

设置 LLM 服务：

```powershell
$env:BALATROLLM_BASE_URL = "https://openrouter.ai/api/v1"
$env:BALATROLLM_API_KEY = "sk-..."
```

先预览任务，不启动游戏：

```powershell
uv run balatrollm config\example.yaml --dry-run
```

正式运行：

```powershell
uv run balatrollm config\example.yaml
```

启动本地可视化页面：

```powershell
uv run balatrollm config\example.yaml --views
```

然后访问：

```text
http://localhost:12345/views/task.html
http://localhost:12345/views/responses.html
```

### 配置方式

配置按以下优先级加载，后者覆盖前者：

1. `BALATROLLM_*` 环境变量
2. YAML 配置文件
3. CLI 参数

可以直接传入配置文件：

```powershell
uv run balatrollm config\example.yaml
```

也可以通过环境变量指定：

```powershell
$env:BALATROLLM_CONFIG = "config\example.yaml"
uv run balatrollm
```

常用参数：

| 参数         | 环境变量              | 默认值      | 作用                |
| ------------ | --------------------- | ----------- | ------------------- |
| `--model`    | `BALATROLLM_MODEL`    | 必填        | LLM 模型名          |
| `--seed`     | `BALATROLLM_SEED`     | `AAAAAAA`   | 游戏种子            |
| `--deck`     | `BALATROLLM_DECK`     | `RED`       | 牌组代码            |
| `--stake`    | `BALATROLLM_STAKE`    | `WHITE`     | 难度代码            |
| `--strategy` | `BALATROLLM_STRATEGY` | `default`   | 策略目录            |
| `--parallel` | `BALATROLLM_PARALLEL` | `1`         | 并发游戏实例数      |
| `--host`     | `BALATROLLM_HOST`     | `127.0.0.1` | BalatroBot 地址     |
| `--port`     | `BALATROLLM_PORT`     | `12346`     | 起始端口            |
| `--base-url` | `BALATROLLM_BASE_URL` | OpenRouter  | LLM API 地址        |
| `--api-key`  | `BALATROLLM_API_KEY`  | 未设置      | LLM API key         |
| `--views`    | `BALATROLLM_VIEWS`    | `0`         | 启动本地 views 服务 |

`model`、`seed`、`deck`、`stake`、`strategy` 都支持多个值，程序会生成笛卡尔积任务：

```powershell
uv run balatrollm --model openai/gpt-4o --deck RED BLUE --seed AAAAAAA BBBBBBB --dry-run
```

### 策略系统

策略位于 `src/balatrollm/strategies/`。内置策略包括：

- `default`
- `aggressive`
- `conservative`

每个策略目录至少需要：

```text
STRATEGY.md.jinja
GAMESTATE.md.jinja
MEMORY.md.jinja
TOOLS.json
```

`STRATEGY.md.jinja` 定义长期打法，`GAMESTATE.md.jinja` 负责渲染当前游戏状态，
`MEMORY.md.jinja` 提供最近动作和错误上下文，`TOOLS.json` 定义不同游戏阶段允许的工具调用。
在选牌阶段，bot 还可以提供只读的 `score_candidates` 观察工具，让模型在真正出牌前比较
多个候选打法。

### 运行产物

每次运行都会写入 `runs/`：

```text
runs/
  latest.json
  v<version>/<strategy>/<vendor>/<model>/<timestamp>_<deck>_<stake>_<seed>/
    task.json
    strategy.json
    run.log
    requests.jsonl
    responses.jsonl
    gamestates.jsonl
    stats.json
    screenshots/
```

这些文件可以用来复盘 prompt、检查模型输出、定位失败工具调用，以及比较不同策略或模型。

### 项目结构

```text
config/                  YAML 配置样例
docs/                    MkDocs 文档
scripts/                 本地辅助脚本
src/balatrollm/          Python 包和 CLI 实现
src/balatrollm/strategies/
tests/                   单元测试
views/                   查看最新运行结果的本地 HTML 页面
```

如果后续要把本地 BalatroBot mod 代码也放进这个仓库，建议保持为单独的顶层目录，
例如 `balatrobot/` 或 `mods/balatrobot/`，不要和 Python 包代码混在一起。

### 开发命令

```powershell
uv sync
uv run pytest
uv run balatrollm --help
```

本地密钥和生成文件不要提交到仓库。API key 放在环境变量或本地忽略文件中，
`runs/`、`logs/`、`.venv/` 和本地配置文件都应保持在版本控制之外。
