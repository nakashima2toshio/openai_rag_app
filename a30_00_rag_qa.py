#
# --------------------------------------------------
# ① カスタマーサポート・FAQデータセット   推奨データセット： Amazon_Polarity
# ② 一般知識・トリビアQAデータセット      推奨データセット： trivia_qa
# ③ 医療質問回答データセット             推奨データセット： FreedomIntelligence/medical-o1-reasoning-SFT
# ④ 科学・技術QAデータセット             推奨データセット： sciq
# ⑤ 法律・判例QAデータセット             推奨データセット： nguha/legalbench
# --------------------------------------------------
# ①　Embeddingの前処理：　1行1ベクトルになる形が理想
# --------------------------------------------------

import os
import re
import time
import json
from pathlib import Path
from typing import List, Optional
import logging

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionSystemMessageParam, \
    ChatCompletionUserMessageParam, ChatCompletionAssistantMessageParam

from datasets import load_dataset
import pandas as pd
import tempfile
import textwrap
from pydantic import BaseModel, Field
from tqdm import tqdm

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ヘルパーモジュールをインポート
try:
    from helper_st import (
        UIHelper, MessageManagerUI, ResponseProcessorUI,
        SessionStateManager, error_handler_ui, timer_ui,
        init_page, select_model, InfoPanelManager
    )
    from helper_api import (
        config, logger, TokenManager, OpenAIClient,
        EasyInputMessageParam, ResponseInputTextParam,
        ConfigManager, MessageManager, sanitize_key,
        error_handler, timer
    )
except ImportError as e:
    st.error(f"ヘルパーモジュールのインポートに失敗しました: {e}")
    st.stop()

BASE_DIR = Path(__file__).resolve().parent.parent       # Paslib
THIS_DIR = Path(__file__).resolve().parent              # Paslib

# ----------------------------------------------------
# ① カスタマーサポート・FAQデータセット 推奨データセット： Amazon_Polarity
# Customer Support FAQ
#  列： question, answer
# 7-16-1: Vector Store ID: vs_68775be00d84819192ecc2b9c1039b89
# ----------------------------------------------------

# ----------------------------------------------------
# ② 一般知識・トリビアQAデータセット: 推奨データセット： trivia_qa
# trivia_qa (13.3GN)
# 列: Question,Complex_CoT,Response
# ----------------------------------------------------

# ----------------------------------------------------
# ③ 医療質問回答データセット: 推奨データセット： FreedomIntelligence/medical-o1-reasoning-SFT
# medical_qa.csv
#  列: Question, Complex_CoT, Response
# ----------------------------------------------------


