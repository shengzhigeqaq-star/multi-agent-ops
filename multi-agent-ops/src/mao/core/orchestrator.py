"""Central Orchestrator — drives multi-agent collaboration with long-chain reasoning.

Core flow (long-chain reasoning pipeline):
  User Input → Planner(decompose) → Executor(act+tool) → Reviewer(verify)
     ↑                                                    │
     └──────────── REVISE loop (deep reasoning) ←─────────┘
                                │ PASS
                                ▼
                          Coordinator(synthesize) → Final Output

Key mechanisms:
  1. Long-chain reasoning: Plan→Execute→Review→Revise→Re-execute→Re-review→Synthesize
  2. Multi-agent collaboration: 4 roles communicate via Message bus
  3. Review-revise cycles: Up to N rounds of quality improvement per subtask
  4. Parallel execution: Independent subtasks run concurrently via DAG topology
  5. Shared context: AgentContext accumulates insights across the reasoning chain
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from mao.agents.coordinator import CoordinatorAgent
from mao.agents.executor import ExecutorAgent
from mao.agents.planner import PlannerAgent
from mao.agents.reviewer import ReviewerAgent
from mao.core.agent import AgentContext
from mao.core.types import (
    AgentConfig,
    AgentRole,
    Message,
    MessageType,
    ProviderType,
    SubTask,
    SystemConfig,
    Task,
    TaskStatus,
)
from mao.llm import create_llm
from mao.memory.sqlite_store import SQLiteStore
from mao.tools.registry import ToolRegistry


@dataclass
class ReasoningStep:
    """Single step in the long-chain reasoning pipeline, preserved for audit."""
    step_id: str
    agent: str
    action: str  # "plan" | "execute" | "review" | "revise" | "arbitrate" | "synthesize"
    subtask_id: str | None = None
    input_summary: str = ""
    output_summary: str = ""
    tool_calls: list[str] = field(default_factory=list)
    verdict: str | None = None  # PASS / REVISE / ESCALATE
    score: int | None = None
    timestamp: str = ""


class Orchestrator:
    """Central orchestrator for multi-agent task execution with long-chain reasoning."""

    def __init__(self, config: SystemConfig):
        self.config = config
        self._store = SQLiteStore(config.storage.get("db_path", "./data/mao.db"))
        self._agents: dict[str, Any] = {}
        self._tool_registry = ToolRegistry(config.tools)

    def _build_agent(self, role: str, agent_cfg: AgentConfig):
        api_key = ""
        if agent_cfg.provider == ProviderType.ANTHROPIC:
            api_key = self.config.api_keys.get("anthropic", "")
        elif agent_cfg.provider == ProviderType.OPENAI:
            api_key = self.config.api_keys.get("openai", "")
        else:
            api_key = self.config.api_keys.get(role, "") or self.config.api_keys.get("openai", "")

        llm = create_llm(
            provider=agent_cfg.provider,
            model=agent_cfg.model,
            api_key=api_key,
            api_base=agent_cfg.api_base,
        )

        agent_tools = ToolRegistry({
            name: self.config.tools.get(name, {"enabled": True})
            for name in agent_cfg.tools
        }) if agent_cfg.tools else self._tool_registry

        role_map = {
            AgentRole.PLANNER: PlannerAgent,
            AgentRole.EXECUTOR: ExecutorAgent,
            AgentRole.REVIEWER: ReviewerAgent,
            AgentRole.COORDINATOR: CoordinatorAgent,
        }

        agent_cls = role_map.get(agent_cfg.role, ExecutorAgent)
        return agent_cls(
            llm=llm,
            tool_registry=agent_tools,
            system_prompt=agent_cfg.system_prompt,
            temperature=agent_cfg.temperature,
            max_tokens=agent_cfg.max_tokens,
        )

    def _get_agents(self) -> dict[str, Any]:
        if not self._agents:
            for key, cfg in self.config.agents.items():
                self._agents[key] = self._build_agent(key, cfg)
        return self._agents

    async def run(
        self,
        description: str,
        workflow_name: str | None = None,
        deep_reasoning: bool = False,
    ) -> Task:
        """Execute a task through multi-agent collaboration.

        Args:
            description: User's task description
            workflow_name: Optional pre-defined workflow template name
            deep_reasoning: If True, enable multi-round review-revise cycles (长链深度推理)
        """
        task = Task(description=description)
        ctx = AgentContext(
            task_id=task.id,
            max_rounds=self.config.defaults.get("max_rounds", 20),
        )
        ctx.shared_memory["deep_reasoning"] = deep_reasoning

        agents = self._get_agents()
        for agent in agents.values():
            agent.set_context(ctx)

        reasoning_chain: list[ReasoningStep] = []
        ctx.shared_memory["reasoning_chain"] = reasoning_chain

        ctx.add_log(f"[Orchestrator] 开始执行任务: {description[:100]}")
        ctx.add_log(f"[Orchestrator] 深度推理模式: {'开启' if deep_reasoning else '标准'}")

        try:
            if workflow_name and workflow_name in self.config.workflows:
                await self._run_workflow(task, agents, ctx, workflow_name, deep_reasoning)
            else:
                await self._run_collaborative(task, agents, ctx, deep_reasoning)

            task.status = TaskStatus.COMPLETED
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.final_output = f"执行失败: {e}"
            ctx.add_log(f"[Orchestrator] 任务失败: {e}")

        task.total_tokens = ctx.total_tokens
        task.rounds = ctx.round
        task.stats = {
            "log": ctx.log,
            "agent_interactions": len(task.messages),
            "reasoning_chain": [
                {
                    "step_id": rs.step_id,
                    "agent": rs.agent,
                    "action": rs.action,
                    "subtask_id": rs.subtask_id,
                    "input": rs.input_summary,
                    "output": rs.output_summary,
                    "tool_calls": rs.tool_calls,
                    "verdict": rs.verdict,
                    "score": rs.score,
                }
                for rs in (ctx.shared_memory.get("reasoning_chain") or [])
            ],
            "deep_reasoning": deep_reasoning,
        }
        task.updated_at = datetime.now(timezone.utc)

        self._store.save_task({
            "id": task.id,
            "description": task.description,
            "status": task.status.value,
            "subtasks": [st.model_dump() for st in task.subtasks],
            "final_output": task.final_output,
            "stats": task.stats,
            "total_tokens": task.total_tokens,
            "rounds": task.rounds,
            "created_at": str(task.created_at),
            "updated_at": str(task.updated_at),
        })

        return task

    # ── Long-Chain Collaborative Execution ─────────────────────────────────

    async def _run_collaborative(
        self,
        task: Task,
        agents: dict[str, Any],
        ctx: AgentContext,
        deep_reasoning: bool = False,
    ) -> None:
        """Execute the full long-chain reasoning pipeline:

        Phase 1: PLAN     — Planner decomposes goal into subtasks
        Phase 2: EXECUTE  — Executor(s) run subtasks (parallel if no deps)
        Phase 3: REVIEW   — Reviewer checks quality of each result
        Phase 4: REVISE   — If rejected, Executor redoes with reviewer feedback (loop)
        Phase 5: ARBITRATE— Coordinator resolves escalated disputes
        Phase 6: SYNTHESIZE— Coordinator integrates all results into final output
        """
        planner = agents.get("planner")
        executor = agents.get("executor")
        reviewer = agents.get("reviewer")
        coordinator = agents.get("coordinator")

        if not all([planner, executor, reviewer, coordinator]):
            raise RuntimeError("Missing required agents: planner, executor, reviewer, coordinator")

        review_required = self.config.defaults.get("review_required", True)
        max_rounds = self.config.defaults.get("max_rounds", 20)
        max_revise_rounds = 3 if deep_reasoning else 1

        chain = ctx.shared_memory.get("reasoning_chain") or []

        # ═══ Phase 1: PLANNING (长链第1步: 任务分解) ═══
        ctx.round = 1
        ctx.add_log("═══ Phase 1: [PLAN] 规划师进行任务分解 ═══")
        plan_msg = Message(
            from_agent="orchestrator",
            to_agent="planner",
            type=MessageType.TASK,
            payload={"description": task.description},
        )
        task.messages.append(plan_msg)
        plan_response = await planner.handle_message(plan_msg)
        task.messages.append(plan_response)

        plan = plan_response.payload.get("plan", {})
        subtask_data = plan.get("subtasks", [])
        strategy = plan.get("strategy", "")

        chain.append(ReasoningStep(
            step_id=f"L1_plan",
            agent="planner",
            action="plan",
            input_summary=task.description[:200],
            output_summary=f"分解为{len(subtask_data)}个子任务. 策略: {strategy[:200]}",
            timestamp=str(datetime.now(timezone.utc)),
        ))
        ctx.shared_memory["reasoning_chain"] = chain
        ctx.add_log(f"[PLAN] 策略: {strategy[:200]}")
        ctx.add_log(f"[PLAN] 子任务数: {len(subtask_data)}")

        # ═══ Phase 2: Build Subtask objects ═══
        ctx.add_log(f"═══ Phase 2: [BUILD] 创建 {len(subtask_data)} 个子任务节点 ═══")
        for sd in subtask_data:
            role_str = sd.get("assigned_role", "executor")
            try:
                assigned_role = AgentRole(role_str)
            except ValueError:
                assigned_role = AgentRole.EXECUTOR

            task.subtasks.append(SubTask(
                id=sd.get("id", ""),
                title=sd.get("title", ""),
                description=sd.get("description", ""),
                assigned_role=assigned_role,
                dependencies=sd.get("dependencies", []),
                expected_output=sd.get("expected_output", ""),
                success_criteria=sd.get("success_criteria", ""),
            ))

        # ═══ Phase 3-4: EXECUTE + REVIEW + REVISE (长链核心循环) ═══
        ctx.add_log("═══ Phase 3-4: [EXECUTE→REVIEW→REVISE] 长链推理核心循环 ═══")
        completed_ids: set[str] = set()
        chain_step_counter = 1

        for round_num in range(1, max_rounds + 1):
            ctx.round = round_num
            ready = [
                st for st in task.subtasks
                if st.id not in completed_ids
                and all(d in completed_ids for d in st.dependencies)
            ]
            if not ready:
                break

            max_parallel = self.config.defaults.get("max_parallel_agents", 4)
            for i in range(0, len(ready), max_parallel):
                batch = ready[i:i + max_parallel]
                batch_tasks = []

                for st in batch:
                    role_key = st.assigned_role.value
                    agent = agents.get(role_key, executor)
                    ctx.add_log(f"[EXECUTE] {agent.name} 开始执行: {st.title}")
                    batch_tasks.append(agent.execute_subtask(st))

                results = await asyncio.gather(*batch_tasks, return_exceptions=True)

                for st, result in zip(batch, results):
                    if isinstance(result, Exception):
                        ctx.add_log(f"[EXECUTE] 子任务 {st.title} 异常: {result}")
                        st.status = TaskStatus.FAILED
                        st.error = str(result)
                        continue

                    completed_ids.add(st.id)
                    chain_step_counter += 1
                    chain.append(ReasoningStep(
                        step_id=f"L{chain_step_counter}_execute",
                        agent=st.assigned_role.value,
                        action="execute",
                        subtask_id=st.id,
                        input_summary=st.description[:200],
                        output_summary=(st.result or "")[:300],
                        tool_calls=[],
                        timestamp=str(datetime.now(timezone.utc)),
                    ))
                    ctx.shared_memory["reasoning_chain"] = chain

                    # ── Review + Revise loop (长链推理的关键: 审核-修改循环) ──
                    if review_required and st.assigned_role == AgentRole.EXECUTOR:
                        for revise_round in range(1, max_revise_rounds + 1):
                            chain_step_counter += 1
                            ctx.add_log(f"[REVIEW] 审核员审查: {st.title} (第{revise_round}轮)")

                            try:
                                st = await reviewer.execute_subtask(st)
                            except Exception as e:
                                ctx.add_log(f"[REVIEW] 审核异常: {e}")
                                break

                            chain.append(ReasoningStep(
                                step_id=f"L{chain_step_counter}_review",
                                agent="reviewer",
                                action="review",
                                subtask_id=st.id,
                                input_summary=(st.result or "")[:200],
                                output_summary=f"结论: {st.review_verdict} 评分: {st.review_score}",
                                verdict=st.review_verdict,
                                score=st.review_score,
                                timestamp=str(datetime.now(timezone.utc)),
                            ))
                            ctx.shared_memory["reasoning_chain"] = chain

                            if st.review_verdict == "PASS":
                                ctx.add_log(f"[REVIEW] ✅ {st.title} 审核通过 (评分: {st.review_score}/10)")
                                break

                            # REVISE: 审核不通过，进入修改循环
                            ctx.add_log(f"[REVISE] ❌ {st.title} 需要修改 (评分: {st.review_score}/10)")
                            ctx.add_log(f"[REVISE] 修改建议: {st.error}")

                            # Deep reasoning: Coordinator arbitrates on repeated failures
                            if deep_reasoning and revise_round >= 2:
                                chain_step_counter += 1
                                ctx.add_log(f"[ARBITRATE] 协调员介入仲裁: {st.title}")
                                arb_result = await coordinator.think(
                                    f"子任务 '{st.title}' 已经修改了{revise_round}轮仍然未通过审核。\n"
                                    f"原始要求: {st.description}\n"
                                    f"当前结果: {st.result}\n"
                                    f"审核意见: {st.error}\n"
                                    f"请做出决策：1) 给执行者更具体的修改指导 2) 降低验收标准 3) 标记为最终失败\n"
                                    f"以JSON输出: {{'decision': 'guide|lower_bar|fail', 'instruction': '...'}}"
                                )
                                ctx.add_log(f"[ARBITRATE] 协调员决策: {arb_result.content[:200]}")
                                chain.append(ReasoningStep(
                                    step_id=f"L{chain_step_counter}_arbitrate",
                                    agent="coordinator",
                                    action="arbitrate",
                                    subtask_id=st.id,
                                    input_summary=f"第{revise_round}轮修改仍未通过",
                                    output_summary=arb_result.content[:300],
                                    timestamp=str(datetime.now(timezone.utc)),
                                ))
                                ctx.shared_memory["reasoning_chain"] = chain

                            # Re-execute with review feedback
                            if revise_round < max_revise_rounds:
                                chain_step_counter += 1
                                ctx.add_log(f"[REVISE] 执行者根据反馈重新执行: {st.title}")
                                st.status = TaskStatus.EXECUTING
                                try:
                                    st = await executor.execute_subtask(st)
                                    chain.append(ReasoningStep(
                                        step_id=f"L{chain_step_counter}_revise",
                                        agent="executor",
                                        action="revise",
                                        subtask_id=st.id,
                                        input_summary=f"修改意见: {(st.error or '')[:200]}",
                                        output_summary=(st.result or "")[:300],
                                        timestamp=str(datetime.now(timezone.utc)),
                                    ))
                                    ctx.shared_memory["reasoning_chain"] = chain
                                except Exception as e:
                                    ctx.add_log(f"[REVISE] 重新执行异常: {e}")
                                    break
                            else:
                                ctx.add_log(f"[REVISE] 已超过最大修改轮次({max_revise_rounds}), 保留当前最佳版本")

                        if st.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                            st.status = TaskStatus.COMPLETED

        # ═══ Phase 5: SYNTHESIS (长链最后一步: 综合输出) ═══
        ctx.round += 1
        ctx.add_log("═══ Phase 5: [SYNTHESIZE] 协调员综合所有结果生成最终输出 ═══")
        task.final_output = await coordinator.synthesize(task.description, task.subtasks)

        chain.append(ReasoningStep(
            step_id=f"L{ctx.round}_synthesize",
            agent="coordinator",
            action="synthesize",
            input_summary=f"综合{len(task.subtasks)}个子任务结果",
            output_summary=(task.final_output or "")[:300],
            timestamp=str(datetime.now(timezone.utc)),
        ))
        ctx.shared_memory["reasoning_chain"] = chain

        ctx.add_log(f"[SYNTHESIZE] 最终输出已生成, 推理链总步数: {len(chain)}")
        task.status = TaskStatus.COMPLETED

    # ── Workflow Template Execution ────────────────────────────────────────

    async def _run_workflow(
        self,
        task: Task,
        agents: dict[str, Any],
        ctx: AgentContext,
        workflow_name: str,
        deep_reasoning: bool = False,
    ) -> None:
        """Execute a pre-defined workflow template with the long-chain pipeline."""
        from mao.core.workflow import WorkflowEngine

        template = self.config.workflows.get(workflow_name, {})
        if not template:
            raise ValueError(f"Workflow '{workflow_name}' not found")

        engine = WorkflowEngine.from_template(template)
        reviewer = agents.get("reviewer")
        chain = ctx.shared_memory.get("reasoning_chain") or []

        for level_idx, level in enumerate(engine.get_execution_order()):
            ctx.round += 1
            parallel_tasks = []

            for step_id in level:
                step = engine.steps[step_id]
                agent = agents.get(step.agent, agents.get("executor"))
                if not agent:
                    ctx.add_log(f"[WORKFLOW] Agent '{step.agent}' not found for step {step_id}")
                    engine.mark_failed(step_id)
                    continue

                ctx.add_log(f"[WORKFLOW] 执行步骤: {step_id} → {agent.name}")
                st = SubTask(id=step_id, title=step.description, description=step.description)
                parallel_tasks.append((step_id, agent.execute_subtask(st)))

            results = await asyncio.gather(
                *(t[1] for t in parallel_tasks), return_exceptions=True
            )

            for (step_id, _), result in zip(parallel_tasks, results):
                if isinstance(result, Exception):
                    ctx.add_log(f"[WORKFLOW] 步骤 {step_id} 失败: {result}")
                    engine.mark_failed(step_id)
                else:
                    # Review workflow steps if reviewer is available
                    if reviewer and hasattr(result, 'result'):
                        try:
                            result = await reviewer.execute_subtask(result)
                        except Exception:
                            pass
                    engine.mark_completed(step_id)
                    if hasattr(result, 'result'):
                        task.subtasks.append(result)

        coordinator = agents.get("coordinator")
        if coordinator and task.subtasks:
            task.final_output = await coordinator.synthesize(task.description, task.subtasks)
        else:
            task.final_output = "\n\n".join(
                f"## {st.title}\n{st.result or ''}" for st in task.subtasks
            )

    # ── Persistence helpers ────────────────────────────────────────────────

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        return self._store.get_task(task_id)

    def list_tasks(self, limit: int = 20) -> list[dict[str, Any]]:
        return self._store.list_tasks(limit)
