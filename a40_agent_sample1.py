from __future__ import annotations
# a40_agent_sample1.py
# --------------------------------------------------------------------
# pip install git+https://github.com/openai/openai-agents-python.git
# 原因：
# PyPIのパッケージopenai-agents==0.1.0には、openai.agentsは存在しない。
# openai==1.91.0にも現時点でAgent SDKは統合されていない。
from agents import GuardrailFunctionOutput
# 改修方法：
# GitHubのリポジトリから最新のAgent SDKをインストールし、
# 正しいインポート（from agents import Agent）を使用する。
# --------------------------------------------------------------------
# ==============================================================
#  a40_agent_sample1.py  ―  OpenAI Agents SDK サンプル（改訂版）
#  --------------------------------------------------------------
#  ・宿題チェック用 Guardrail エージェント
#  ・Math / History Tutor へのハンドオフ
#  ・InputGuardrailTripwireTriggered 例外を捕捉して Graceful-Fail
# ==============================================================
import asyncio
from pydantic import BaseModel

# ===== OpenAI Agents SDK =====
from agents.agent import Agent
from agents.run import Runner
from agents.run_context import RunContextWrapper
from agents.guardrail import InputGuardrail, GuardrailFunctionOutput
from agents.agent_output import AgentOutputSchema  # 使わない場合は削除可
from agents.tool import function_tool               # 〃
from agents.models.interface import Model           # 〃
from agents.exceptions import InputGuardrailTripwireTriggered

# --------------------------------------------------------------
# 1) Guardrail 用の出力スキーマ
# --------------------------------------------------------------
class HomeworkOutput(BaseModel):
    is_homework: bool
    reasoning: str

# --------------------------------------------------------------
# 2) エージェント定義
# --------------------------------------------------------------
# --- Guardrail チェッカー -------------------------------------
guardrail_agent = Agent(
    name="Guardrail check",
    instructions="Check if the user is asking about homework.",
    output_type=HomeworkOutput,
)

# --- Tutor エージェント群 --------------------------------------
math_tutor_agent = Agent(
    name="Math Tutor",
    handoff_description="Specialist agent for math questions",
    instructions=(
        "You provide help with math problems. "
        "Explain your reasoning at each step and include examples."
    ),
)

history_tutor_agent = Agent(
    name="History Tutor",
    handoff_description="Specialist agent for historical questions",
    instructions=(
        "You provide assistance with historical queries. "
        "Explain important events and context clearly."
    ),
)

# --------------------------------------------------------------
# 3) Guardrail 関数
# --------------------------------------------------------------
async def homework_guardrail(
    ctx: RunContextWrapper[None], agent: Agent, input_data: str
) -> GuardrailFunctionOutput:
    """
    宿題かどうかを判定し、宿題であれば tripwire を発火させる。
    """
    result = await Runner.run(guardrail_agent, input_data, context=ctx.context)
    final_output: HomeworkOutput = result.final_output_as(HomeworkOutput)

    # tripwire_triggered=True なら Runner が例外を送出して停止する
    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=final_output.is_homework,  # ← 宿題なら True
    )

# --------------------------------------------------------------
# 4) Triage エージェント（Router）
# --------------------------------------------------------------
triage_agent = Agent(
    name="Triage Agent",
    instructions="You determine which agent to use based on the user's homework question",
    handoffs=[history_tutor_agent, math_tutor_agent],
    input_guardrails=[
        InputGuardrail(guardrail_function=homework_guardrail),
    ],
)

# --------------------------------------------------------------
# 5) エントリーポイント
# --------------------------------------------------------------
async def main() -> None:
    """
    2 つのクエリで動作確認：
        ① 宿題ではない歴史の質問          → History Tutor が応答
        ② 哲学的な（宿題扱いする）質問    → Guardrail がブロック
    """
    queries = [
        "who was the first president of the united states?",
        "what is life",
    ]

    for q in queries:
        try:
            result = await Runner.run(triage_agent, q)
            print(f"Query: {q!r}\nAnswer:\n{result.final_output}\n{'-'*60}")
        except InputGuardrailTripwireTriggered:
            print(f"Query: {q!r}\n⚠ Guardrail: 宿題と判定されたため回答を拒否しました。\n{'-'*60}")

# --------------------------------------------------------------
# 6) スクリプト実行
# --------------------------------------------------------------
if __name__ == "__main__":
    asyncio.run(main())

