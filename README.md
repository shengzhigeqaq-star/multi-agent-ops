**Quartet** 是一个多Agent角色协作框架。它通过"规划师-执行者-审核员-协调员"四种角色组成的长链推理流水线，将复杂运营任务自动化，并内置自校正机制确保输出质量。

## 核心思路

单一LLM面对复杂任务时无自校正能力——输出即终点，多步推理后幻觉累积。真实运营场景（报告撰写、数据分析、代码审查）天然需要"策划→执行→审核→汇总"的角色分工。

Quartet 将四种角色固化到系统中，通过 **审核-修改循环** 实现闭环自校正：

```
用户输入 → Planner(分解) → Executor(执行+工具) → Reviewer(审核)
                        ↑                              │
                        └── REVISE 退回修改 ←───────────┘
                                      │
                                 PASS ↓
                              Coordinator(综合) → 最终交付物
```

## 快速开始

```bash
# 1. 安装
git clone https://github.com/your-org/quartet.git
cd quartet
pip install -r requirements.txt

# 2. 配置 API Key
cp .env.example .env
# 编辑 .env，填入 OPENAI_API_KEY / ANTHROPIC_API_KEY 等

# 3. 运行第一个任务
python -m mao.main run "写一份AI Agent市场分析报告"

# 4. 深度推理模式（多轮审核-修改-仲裁）
python -m mao.main run --deep "分析2025年云计算趋势并给出建议"
```

## 命令一览

```bash
mao run <任务描述>           # 启动多Agent协作任务
mao run --deep <任务描述>    # 深度推理模式（3轮审核循环+仲裁）
mao run -f report_generation # 使用预定义工作流
mao status <task_id>         # 查看任务状态
mao list                     # 列出最近任务
mao inspect <task_id>        # 查看执行日志
mao inspect --chain <id>     # 回溯完整长链推理过程
mao agents                   # 列出可用Agent
mao tools                    # 列出可用工具
```

## 架构

```
src/mao/
├── core/          # 编排器、工作流引擎、Agent基类、类型定义
├── agents/        # 规划师 | 执行者 | 审核员 | 协调员
├── llm/           # OpenAI兼容 + Anthropic Claude 双后端
├── tools/         # 文件读写 | 网络搜索 | Shell | Python执行
├── memory/        # SQLite持久化 + ChromaDB向量记忆
├── cli/           # Rich CLI界面
└── config/        # YAML配置驱动
```

四个Agent可独立配置不同模型，例如：Planner用Claude Opus做深度规划，Executor用DeepSeek做高性价比执行，Reviewer用GPT-4o做严谨审核。

## 预定义工作流

| 工作流 | 说明 |
|--------|------|
| `report_generation` | 搜索→撰写→审核→汇总 |
| `data_analysis` | 探索→分析→审核→报告 |
| `code_review` | 多维度并行审查→汇总意见 |
| `collaborative_discussion` | 多Agent多轮深度讨论 |

## 配置

在 `config/agents.yaml` 中自定义Agent行为：

```yaml
planner:
  model: "claude-opus-4-7"    # 规划用强模型
  provider: "anthropic"
  temperature: 0.3             # 低温度保证规划稳定

executor:
  model: "deepseek-chat"       # 执行用性价比模型
  provider: "openai_compatible"
  tools: ["web_search", "web_fetch", "file_write", "python_exec"]
```

## License

MIT
