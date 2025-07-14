# pip install --upgrade git+https://github.com/openai/openai-agents-python.git
# 基本
from agents.agent import Agent
from agents.run import Runner
from agents.run_context import RunContextWrapper

# ツール関連（必要に応じて）
from agents.tool import function_tool
from agents.models.interface import Model

# 出力スキーマ関連（必要に応じて）
from agents.agent_output import AgentOutputSchema
from agents.guardrail import GuardrailFunctionOutput

from pydantic import BaseModel
import asyncio

from agents import Agent

def main():
    agent = Agent(
        name="Hello",
        instructions="任意の質問に答えてください。",
    )
    print(agent)  # オブジェクトが生成されればインストール成功

if __name__ == "__main__":
    main()
