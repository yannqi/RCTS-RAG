import os
import copy
from functools import lru_cache
import asyncio
import json
# import aioboto3
import aiohttp
import numpy as np
# import ollama
from functools import wraps
from openai import (
    AsyncOpenAI,
    APIConnectionError,
    RateLimitError,
    Timeout,
    AsyncAzureOpenAI,
)

import base64
import struct
try:
    import lmdeploy
    from lmdeploy import version_info, GenerationConfig
    from lmdeploy import pipeline, ChatTemplateConfig, TurbomindEngineConfig
except Exception:
    ImportError("Please install lmdeploy before intialize lmdeploy backend.")
    
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
import torch
from pydantic import BaseModel, Field
from typing import List, Dict, Callable, Any
from module.storage.base import BaseKVStorage
from module.storage import compute_args_hash, wrap_embedding_func_with_attrs
# from vllm import SamplingParams, LLM

os.environ["TOKENIZERS_PARALLELISM"] = "false"


async def vqa_model_func(
        prompt, system_prompt=None, history_messages=[], extra_body=None, **kwargs
        ) -> str:
        # TODO 异步调用
        return await openai_complete_if_cache(  
            model=os.environ.get("VQA_MODEL"), 
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            base_url=os.environ.get("BASE_URL"),
            api_key=os.environ.get("API_KEY"),
            extra_body=extra_body,
            **kwargs,
        )


async def vqa_model_func_with_multi_node(
        prompt, system_prompt=None, node_index=0, history_messages=[], extra_body=None, **kwargs
        ) -> str:
        # TODO 异步调用
    
        return await openai_complete_if_cache(  
            model=os.environ.get("VQA_MODEL"), 
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            base_url=os.environ.get("BASE_URL").split(';')[node_index],
            api_key=os.environ.get("API_KEY"),
            extra_body=extra_body,
            **kwargs,
        )

async def reward_vqa_model_func(
        prompt, system_prompt=None, history_messages=[], extra_body=None, **kwargs
        ) -> str:
        return await openai_complete_if_cache(  
            model=os.environ.get("REWARD_VQA_MODEL"), 
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            base_url=os.environ.get("REWARD_BASE_URL"),
            api_key=os.environ.get("REWARD_API_KEY"),
            extra_body=extra_body,
            **kwargs,
        )

async def vqa_model_func_vllm_inference(
        prompt, system_prompt=None, history_messages=[], extra_body=None, **kwargs
        ) -> str:
        return await vllm_model_if_cache(  
            model=os.environ.get("VQA_MODEL"), 
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            extra_body=extra_body,
            **kwargs,
        )



def limit_async_func_call(max_size: int, waitting_time: float = 0.0001):
    """Add restriction of maximum async calling times for a async func"""

    def final_decro(func):
        """Not using async.Semaphore to aovid use nest-asyncio"""
        __current_size = 0

        @wraps(func)
        async def wait_func(*args, **kwargs):
            nonlocal __current_size
            while __current_size >= max_size:
                await asyncio.sleep(waitting_time)
            __current_size += 1
            result = await func(*args, **kwargs)
            __current_size -= 1
            return result

        return wait_func

    return final_decro


def limit_async_func_call_with_multi_node(max_size: int, max_node: int, waitting_time: float = 0.1):
    """Add restriction of maximum async calling times for a async func"""

    def final_decro(func):
        """Not using async.Semaphore to aovid use nest-asyncio"""
        __current_sizes = [0] * max_node
 
        @wraps(func)
        async def wait_func(*args, **kwargs):
            nonlocal __current_sizes
            while sum(__current_sizes) >= max_size * max_node:
                await asyncio.sleep(waitting_time)
            node_index = __current_sizes.index(min(__current_sizes))
            __current_sizes[node_index] += 1
            result = await func(node_index=node_index, *args, **kwargs)
            __current_sizes[node_index] -= 1
            return result

        return wait_func

    return final_decro



@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)
async def openai_complete_if_cache(
    model,
    prompt,
    extra_body=None,
    system_prompt=None,
    history_messages=[],
    base_url=None,
    api_key=None,
    **kwargs,
) -> str:
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    openai_async_client = (
        AsyncOpenAI() if base_url is None else AsyncOpenAI(base_url=base_url)
    )
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)

    messages = prompt
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    model_cards = await openai_async_client.models.list()._get_page()
    response = await openai_async_client.chat.completions.create(
        model=model_cards.data[0].id, messages=messages, extra_body=extra_body, **kwargs
    )
 
    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": model}}
        )
    if len(response.choices) > 1:
        return [choice.message.content for choice in response.choices] 
    return response.choices[0].message.content







@lru_cache(maxsize=1)
def initialize_vllm_model(model_name):
    llm = LLM(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=5,
         limit_mm_per_prompt={"image": 10},
        # Note - mm_processor_kwargs can also be passed to generate/chat calls
        mm_processor_kwargs={
            "min_pixels": 224 * 224,
            "max_pixels": 1280 * 28 * 28,
        },
        # mm_cache_preprocessor=args.mm_cache_preprocessor,
    )
    processor = AutoProcessor.from_pretrained(model_name)
    
    return llm


async def vllm_model_if_cache(
    model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    model_name = model
    hf_model, hf_tokenizer = initialize_hf_model(model_name)
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]
    input_prompt = ""
    try:
        input_prompt = hf_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        try:
            ori_message = copy.deepcopy(messages)
            if messages[0]["role"] == "system":
                messages[1]["content"] = (
                    "<system>"
                    + messages[0]["content"]
                    + "</system>\n"
                    + messages[1]["content"]
                )
                messages = messages[1:]
                input_prompt = hf_tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
        except Exception:
            len_message = len(ori_message)
            for msgid in range(len_message):
                input_prompt = (
                    input_prompt
                    + "<"
                    + ori_message[msgid]["role"]
                    + ">"
                    + ori_message[msgid]["content"]
                    + "</"
                    + ori_message[msgid]["role"]
                    + ">\n"
                )

    input_ids = hf_tokenizer(
        input_prompt, return_tensors="pt", padding=True, truncation=True
    ).to("cuda")
    inputs = {k: v.to(hf_model.device) for k, v in input_ids.items()}
    output = hf_model.generate(
        **input_ids, max_new_tokens=512, num_return_sequences=1, early_stopping=True
    )
    response_text = hf_tokenizer.decode(
        output[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True
    )
    if hashing_kv is not None:
        await hashing_kv.upsert({args_hash: {"return": response_text, "model": model}})
    return response_text



@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)
async def azure_openai_complete_if_cache(
    model,
    prompt,
    system_prompt=None,
    history_messages=[],
    base_url=None,
    api_key=None,
    **kwargs,
):
    if api_key:
        os.environ["AZURE_OPENAI_API_KEY"] = api_key
    if base_url:
        os.environ["AZURE_OPENAI_ENDPOINT"] = base_url

    openai_async_client = AsyncAzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )

    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    if prompt is not None:
        messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    response = await openai_async_client.chat.completions.create(
        model=model, messages=messages, **kwargs
    )

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": model}}
        )
    return response.choices[0].message.content


class BedrockError(Exception):
    """Generic error for issues related to Amazon Bedrock"""


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, max=60),
    retry=retry_if_exception_type((BedrockError)),
)
async def bedrock_complete_if_cache(
    model,
    prompt,
    system_prompt=None,
    history_messages=[],
    aws_access_key_id=None,
    aws_secret_access_key=None,
    aws_session_token=None,
    **kwargs,
) -> str:
    os.environ["AWS_ACCESS_KEY_ID"] = os.environ.get(
        "AWS_ACCESS_KEY_ID", aws_access_key_id
    )
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ.get(
        "AWS_SECRET_ACCESS_KEY", aws_secret_access_key
    )
    os.environ["AWS_SESSION_TOKEN"] = os.environ.get(
        "AWS_SESSION_TOKEN", aws_session_token
    )

    # Fix message history format
    messages = []
    for history_message in history_messages:
        message = copy.copy(history_message)
        message["content"] = [{"text": message["content"]}]
        messages.append(message)

    # Add user prompt
    messages.append({"role": "user", "content": [{"text": prompt}]})

    # Initialize Converse API arguments
    args = {"modelId": model, "messages": messages}

    # Define system prompt
    if system_prompt:
        args["system"] = [{"text": system_prompt}]

    # Map and set up inference parameters
    inference_params_map = {
        "max_tokens": "maxTokens",
        "top_p": "topP",
        "stop_sequences": "stopSequences",
    }
    if inference_params := list(
        set(kwargs) & set(["max_tokens", "temperature", "top_p", "stop_sequences"])
    ):
        args["inferenceConfig"] = {}
        for param in inference_params:
            args["inferenceConfig"][inference_params_map.get(param, param)] = (
                kwargs.pop(param)
            )

    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    # Call model via Converse API
    session = aioboto3.Session()
    async with session.client("bedrock-runtime") as bedrock_async_client:
        try:
            response = await bedrock_async_client.converse(**args, **kwargs)
        except Exception as e:
            raise BedrockError(e)

        if hashing_kv is not None:
            await hashing_kv.upsert(
                {
                    args_hash: {
                        "return": response["output"]["message"]["content"][0]["text"],
                        "model": model,
                    }
                }
            )

        return response["output"]["message"]["content"][0]["text"]


@lru_cache(maxsize=1)
def initialize_hf_model(model_name):
    hf_tokenizer = AutoTokenizer.from_pretrained(
        model_name, device_map="auto", trust_remote_code=True
    )
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", trust_remote_code=True
    )
    if hf_tokenizer.pad_token is None:
        hf_tokenizer.pad_token = hf_tokenizer.eos_token

    return hf_model, hf_tokenizer


async def hf_model_if_cache(
    model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    model_name = model
    hf_model, hf_tokenizer = initialize_hf_model(model_name)
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]
    input_prompt = ""
    try:
        input_prompt = hf_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        try:
            ori_message = copy.deepcopy(messages)
            if messages[0]["role"] == "system":
                messages[1]["content"] = (
                    "<system>"
                    + messages[0]["content"]
                    + "</system>\n"
                    + messages[1]["content"]
                )
                messages = messages[1:]
                input_prompt = hf_tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
        except Exception:
            len_message = len(ori_message)
            for msgid in range(len_message):
                input_prompt = (
                    input_prompt
                    + "<"
                    + ori_message[msgid]["role"]
                    + ">"
                    + ori_message[msgid]["content"]
                    + "</"
                    + ori_message[msgid]["role"]
                    + ">\n"
                )

    input_ids = hf_tokenizer(
        input_prompt, return_tensors="pt", padding=True, truncation=True
    ).to("cuda")
    inputs = {k: v.to(hf_model.device) for k, v in input_ids.items()}
    output = hf_model.generate(
        **input_ids, max_new_tokens=512, num_return_sequences=1, early_stopping=True
    )
    response_text = hf_tokenizer.decode(
        output[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True
    )
    if hashing_kv is not None:
        await hashing_kv.upsert({args_hash: {"return": response_text, "model": model}})
    return response_text


async def ollama_model_if_cache(
    model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    kwargs.pop("max_tokens", None)
    kwargs.pop("response_format", None)
    host = kwargs.pop("host", None)
    timeout = kwargs.pop("timeout", None)

    ollama_client = ollama.AsyncClient(host=host, timeout=timeout)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    response = await ollama_client.chat(model=model, messages=messages, **kwargs)

    result = response["message"]["content"]

    if hashing_kv is not None:
        await hashing_kv.upsert({args_hash: {"return": result, "model": model}})

    return result


@lru_cache(maxsize=1)
def initialize_lmdeploy_pipeline(
    model,
    tp=1,
    chat_template=None,
    log_level="WARNING",
    model_format="hf",
    quant_policy=0,
):
    

    lmdeploy_pipe = pipeline(
        model_path=model,
        backend_config=TurbomindEngineConfig(
            tp=tp, model_format=model_format, quant_policy=quant_policy
        ),
        chat_template_config=ChatTemplateConfig(model_name=chat_template)
        if chat_template
        else None,
        log_level="WARNING",
    )
    return lmdeploy_pipe


async def lmdeploy_model_if_cache(
    lmdeploy_pipe,
    messages,
    extra_api_key = {},
    **kwargs,
) -> str:
    """
    Args:
        model (str): The path to the model.
            It could be one of the following options:
                    - i) A local directory path of a turbomind model which is
                        converted by `lmdeploy convert` command or download
                        from ii) and iii).
                    - ii) The model_id of a lmdeploy-quantized model hosted
                        inside a model repo on huggingface.co, such as
                        "InternLM/internlm-chat-20b-4bit",
                        "lmdeploy/llama2-chat-70b-4bit", etc.
                    - iii) The model_id of a model hosted inside a model repo
                        on huggingface.co, such as "internlm/internlm-chat-7b",
                        "Qwen/Qwen-7B-Chat ", "baichuan-inc/Baichuan2-7B-Chat"
                        and so on.
        chat_template (str): needed when model is a pytorch model on
            huggingface.co, such as "internlm-chat-7b",
            "Qwen-7B-Chat ", "Baichuan2-7B-Chat" and so on,
            and when the model name of local path did not match the original model name in HF.
        tp (int): tensor parallel
        prompt (Union[str, List[str]]): input texts to be completed.
        do_preprocess (bool): whether pre-process the messages. Default to
            True, which means chat_template will be applied.
        skip_special_tokens (bool): Whether or not to remove special tokens
            in the decoding. Default to be True.
        do_sample (bool): Whether or not to use sampling, use greedy decoding otherwise.
            Default to be False, which means greedy decoding will be applied.
    """
    skip_special_tokens = extra_api_key.pop("skip_special_tokens", False)
    max_new_tokens = extra_api_key.pop("max_new_tokens", 512)
    temperature = extra_api_key.pop("temperature", 1.0)
    top_p = extra_api_key.pop("top_p", 1.0)
    repetition_penalty = extra_api_key.pop("repetition_penalty", 1.0)
    max_tokens = extra_api_key.pop("max_tokens", 512)


  
   


    gen_config = GenerationConfig(
        skip_special_tokens=skip_special_tokens,
        max_new_tokens=max_new_tokens,
        **gen_params,
    )

    response = ""
    async for res in lmdeploy_pipe.generate(
        messages,
        gen_config=gen_config,
        do_preprocess=do_preprocess,
        stream_response=False,
        session_id=1,
    ):
        response += res.response

    return response


async def gpt_4o_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        "gpt-4o",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def gpt_4o_mini_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        "gpt-4o-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def azure_openai_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await azure_openai_complete_if_cache(
        "conversation-4o-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def bedrock_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await bedrock_complete_if_cache(
        "anthropic.claude-3-haiku-20240307-v1:0",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def hf_model_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    model_name = kwargs["hashing_kv"].global_config["llm_model_name"]
    return await hf_model_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def ollama_model_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    model_name = kwargs["hashing_kv"].global_config["llm_model_name"]
    return await ollama_model_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )




class Model(BaseModel):
    """
    This is a Pydantic model class named 'Model' that is used to define a custom language model.

    Attributes:
        gen_func (Callable[[Any], str]): A callable function that generates the response from the language model.
            The function should take any argument and return a string.
        kwargs (Dict[str, Any]): A dictionary that contains the arguments to pass to the callable function.
            This could include parameters such as the model name, API key, etc.

    Example usage:
        Model(gen_func=openai_complete_if_cache, kwargs={"model": "gpt-4", "api_key": os.environ["OPENAI_API_KEY_1"]})

    In this example, 'openai_complete_if_cache' is the callable function that generates the response from the OpenAI model.
    The 'kwargs' dictionary contains the model name and API key to be passed to the function.
    """

    gen_func: Callable[[Any], str] = Field(
        ...,
        description="A function that generates the response from the llm. The response must be a string",
    )
    kwargs: Dict[str, Any] = Field(
        ...,
        description="The arguments to pass to the callable function. Eg. the api key, model name, etc",
    )

    class Config:
        arbitrary_types_allowed = True


class MultiModel:
    """
    Distributes the load across multiple language models. Useful for circumventing low rate limits with certain api providers especially if you are on the free tier.
    Could also be used for spliting across diffrent models or providers.

    Attributes:
        models (List[Model]): A list of language models to be used.

    Usage example:
        ```python
        models = [
            Model(gen_func=openai_complete_if_cache, kwargs={"model": "gpt-4", "api_key": os.environ["OPENAI_API_KEY_1"]}),
            Model(gen_func=openai_complete_if_cache, kwargs={"model": "gpt-4", "api_key": os.environ["OPENAI_API_KEY_2"]}),
            Model(gen_func=openai_complete_if_cache, kwargs={"model": "gpt-4", "api_key": os.environ["OPENAI_API_KEY_3"]}),
            Model(gen_func=openai_complete_if_cache, kwargs={"model": "gpt-4", "api_key": os.environ["OPENAI_API_KEY_4"]}),
            Model(gen_func=openai_complete_if_cache, kwargs={"model": "gpt-4", "api_key": os.environ["OPENAI_API_KEY_5"]}),
        ]
        multi_model = MultiModel(models)
        rag = LightRAG(
            llm_model_func=multi_model.llm_model_func
            / ..other args
            )
        ```
    """

    def __init__(self, models: List[Model]):
        self._models = models
        self._current_model = 0

    def _next_model(self):
        self._current_model = (self._current_model + 1) % len(self._models)
        return self._models[self._current_model]

    async def llm_model_func(
        self, prompt, system_prompt=None, history_messages=[], **kwargs
    ) -> str:
        kwargs.pop("model", None)  # stop from overwriting the custom model name
        next_model = self._next_model()
        args = dict(
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs,
            **next_model.kwargs,
        )

        return await next_model.gen_func(**args)


if __name__ == "__main__":
    import asyncio

    async def main():
        result = await gpt_4o_mini_complete("How are you?")
        print(result)

    asyncio.run(main())