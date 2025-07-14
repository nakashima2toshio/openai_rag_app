#
from openai.agents import Agent

from typing import Any
from pydantic import BaseModel
import asyncio

# 正しいインポート文（実際のパッケージ名に応じて調整してください）
from agents import Agent, InputGuardrail, GuardrailFunctionOutput, Runner, RunContextWrapper

class HomeworkOutput(BaseModel):
    is_homework: bool
    reasoning: str

# エージェントの定義（型パラメータを削除）
guardrail_agent = Agent(
    name="Guardrail check",
    instructions="Check if the user is asking about homework.",
    output_type=HomeworkOutput,
)

math_tutor_agent = Agent(
    name="Math Tutor",
    handoff_description="Specialist agent for math questions",
    instructions="You provide help with math problems. Explain your reasoning at each step and include examples",
)

history_tutor_agent = Agent(
    name="History Tutor",
    handoff_description="Specialist agent for historical questions",
    instructions="You provide assistance with historical queries. Explain important events and context clearly.",
)

# guardrail関数（tripwireのロジックも修正）
async def homework_guardrail(ctx: RunContextWrapper[dict], agent: Agent, input_data: str) -> GuardrailFunctionOutput:
    result = await Runner.run(guardrail_agent, input_data, context=ctx.context)
    final_output = result.final_output_as(HomeworkOutput)
    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=final_output.is_homework,  # 修正：ホームワークの場合にtripwireを発動
    )

# InputGuardrailの型パラメータを削除
triage_agent = Agent(
    name="Triage Agent",
    instructions="You determine which agent to use based on the user's homework question",
    handoffs=[history_tutor_agent, math_tutor_agent],
    input_guardrails=[
        InputGuardrail(guardrail_function=homework_guardrail),  # 修正済
    ],
)

async def main():
    result = await Runner.run(triage_agent, "who was the first president of the united states?", context={})
    print(result.final_output)

    result = await Runner.run(triage_agent, "what is life", context={})
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
