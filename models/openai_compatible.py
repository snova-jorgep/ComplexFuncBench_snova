from typing import Any, Dict
import os
from openai import OpenAI
import json
import sys
import copy
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompts.prompts import SimpleTemplatePrompt
from utils.utils import *
from dotenv import load_dotenv
import yaml

load_dotenv()


class OpenAICompatibleModel:
    def __init__(self, model_name):
        super().__init__()
        with open("models/config/config.yaml", "r") as yaml_file:
            self.config = yaml.safe_load(yaml_file)
        self.base_urls = self.config.get("urls")
        self.model_mappings = self.config.get("model_mappings")
        self.max_retires = self.config.get("max_retires")
        self.timeout = self.config.get("timeout")
        self.client, self.model_name = self._init_client(model_name)
        

    def __call__(self, prefix, prompt: SimpleTemplatePrompt, **kwargs: Any):
        filled_prompt = prompt(**kwargs)
        prediction = self._predict(prefix, filled_prompt, **kwargs)
        return prediction
    
    @retry(max_attempts=10)
    def _predict(self, prefix, text, **kwargs):
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": prefix},
                    {"role": "user", "content": text}
                ],
                temperature=0.0,
                )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Exception: {e}")
            return None

    def _init_client(self, model_name):
        split_name = model_name.split("/")
        provider = split_name[0]
        model_name = "/".join(split_name[1:])
        if provider == "sambanova":
            return OpenAI(
                    base_url=self.base_urls["sambanova_base_url"],
                    api_key=os.environ.get("SAMBANOVA_API_KEY", ""),
                    max_retries=self.max_retires,
                    timeout=self.timeout,
                ), model_name
        elif provider == "fireworks":
            return OpenAI(
                base_url=self.base_urls["fireworks_base_url"],
                api_key=os.environ.get("FIREWORKS_API_KEY", ""),
                max_retries=self.max_retires,
                timeout=self.timeout,
            ), model_name
        elif provider == "groq":
            return OpenAI(
                base_url=self.base_urls["groq_base_url"],
                api_key=os.environ.get("GROQ_API_KEY", ""),
                max_retries=self.max_retires,
                timeout=self.timeout,
            ), model_name
        elif provider == "cerebras":
            return OpenAI(
                base_url=self.base_urls["cerebras_base_url"],
                api_key=os.environ.get("CEREBRAS_API_KEY", ""),
                max_retries=self.max_retires,
                timeout=self.timeout,
            ), model_name
        elif provider == "together":
            return OpenAI(
                base_url=self.base_urls["together_base_url"],
                api_key=os.environ.get("TOGETHER_API_KEY", ""),
                max_retries=self.max_retires,
                timeout=self.timeout,
            ), model_name
        else:
            raise ValueError(f"Unsupported provider '{provider}'. Must be one of: sambanova, fireworks, groq, cerebras, together.")


class FunctionCallOpenAICompatible(OpenAICompatibleModel):
    def __init__(self, model_name):
        super().__init__(model_name)
        # self.model_name = model_name
        self.messages = []

    @retry(max_attempts=5, delay=10)
    def __call__(self, messages, tools=None, **kwargs: Any):
        if "function_call" not in json.dumps(messages, ensure_ascii=False):
            self.messages = copy.deepcopy(messages)
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.messages,
                temperature=0.0,
                tools=tools,
                tool_choice="auto",
                max_tokens=2048
            )
            return completion.choices[0].message
        except Exception as e:
            print(f"Exception: {e}")
            return None


if __name__ == "__main__":
    model = OpenAICompatibleModel("sambanova/Meta-Llama-3.3-70B-Instruct")
    response = model("You are a helpful assistant.", SimpleTemplatePrompt(template=("What is the capital of France?"), args_order=[]))
    print(response)