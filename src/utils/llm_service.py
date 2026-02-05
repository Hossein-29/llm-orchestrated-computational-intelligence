import openai
from enum import Enum
from pydantic import BaseModel
import json
import httpx
import os
from dotenv import load_dotenv

load_dotenv()

http_client = httpx.Client(
    timeout=60,
)

client = openai.OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1", http_client=http_client, max_retries=3
)

class Model(Enum):
    GLM_4_7 = "z-ai/glm-4.7"
    GPT_OSS = "openai/gpt-oss-120b"

def get_chat_completion(model: str, messages: list[dict], max_tokens=500) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def get_chat_completion_structured(model: str, messages: list[dict], response_format: BaseModel, max_tokens=500) -> str:
    product_schema = response_format.model_json_schema()

    product_schema.setdefault("additionalProperties", False)

    response_format_dict = {
        "type": "json_schema",
        "json_schema": {"name": response_format.__name__, "strict": True, "schema": product_schema},
    }

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format=response_format_dict,
        temperature=0.0,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content
    data = json.loads(content)
    return response_format(**data)
    