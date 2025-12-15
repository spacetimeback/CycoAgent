import os
import httpx
import json
import subprocess
import logging
from abc import ABC
from typing import Callable, Union, Dict, Any, List
from .utils import build_llama2_prompt
from .exceptions import LongerThanContextError
import aiohttp
import asyncio
import threading
import time

# Set up logging
logger = logging.getLogger(__name__)

### when use tgi model
api_key = '-' 


class ChatOpenAICompatible(ABC):
    def __init__(
        self,
        end_point: str = "https://api.nuwaapi.com/v1/chat/completions",
        model="gemini-2.5-pro-exp-03-25",
        system_message: str = "You are a helpful assistant.",
        other_parameters: Union[Dict[str, Any], None] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ):
        try:
            # Validate inputs
            if not isinstance(end_point, str):
                raise TypeError("end_point must be a string")
            if not isinstance(model, str):
                raise TypeError("model must be a string")
            if not isinstance(system_message, str):
                raise TypeError("system_message must be a string")
            if other_parameters is not None and not isinstance(other_parameters, dict):
                raise TypeError("other_parameters must be a dictionary or None")
            
            # Get API key
            api_key = os.environ.get("OPENAI_API_KEY", "-")
            if api_key == "-":
                logger.warning("No API key found in environment variables")
            
            logger.info(f"Using API key: {api_key[:4]}...")
            logger.info(f"Using endpoint: {end_point}")
            logger.info(f"Using model: {model}")
            
            self.end_point = end_point
            self.model = model
            self.system_message = system_message
            self.other_parameters = other_parameters
            self.temperature = temperature
            self.max_tokens = max_tokens
            
            # Set headers based on model type
            if self.model.startswith("gemini-pro"):
                try:
                    proc_result = subprocess.run(
                        ["gcloud", "auth", "print-access-token"],
                        capture_output=True,
                        text=True,
                        check=True
                    )
                    access_token = proc_result.stdout.strip()
                    self.headers = {
                        "Authorization": f"Bearer {access_token}",
                        "Content-Type": "application/json",
                    }
                except subprocess.CalledProcessError as e:
                    logger.error(f"Error getting Google Cloud access token: {e}")
                    raise
            elif self.model.startswith("tgi"):
                self.headers = {
                    'Content-Type': 'application/json'
                }
            else:
                self.headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }
                self.other_parameters = {} if other_parameters is None else other_parameters
                logger.info(f"Headers: {self.headers}")
                
            self.max_retries = 3
            self.retry_delay = 1.0
            self.timeout = 60.0  # 设置合理的超时时间
            
        except Exception as e:
            logger.error(f"Error initializing ChatOpenAICompatible: {str(e)}")
            raise

    def parse_response(self, response: httpx.Response) -> str:
        try:
            if not isinstance(response, httpx.Response):
                raise TypeError("response must be a httpx.Response instance")
            
            # Convert exp version to preview version if needed
            model_name = self.model
            if "gemini-2.5-pro-exp" in model_name:
                model_name = model_name.replace("exp", "preview")
            
            if model_name.startswith("gpt") or model_name.startswith("deepseek"):
                response_out = response.json()
                logger.info(f"Parsed GPT/Deepseek response: {json.dumps(response_out, indent=2)}")
                if "choices" not in response_out or not response_out["choices"]:
                    raise ValueError("Invalid response format: missing choices")
                return response_out["choices"][0]["message"]["content"]
                
            elif "gemini-2.5-pro" in model_name:
                response_out = response.json()
                logger.info(f"Parsed Gemini response: {json.dumps(response_out, indent=2)}")
                if "choices" not in response_out or not response_out["choices"]:
                    raise ValueError("Invalid response format: missing choices")
                return response_out["choices"][0]["message"]["content"]
                
            elif model_name.startswith("tgi"):
                response_out = response.json()
                logger.info(f"Parsed TGI response: {json.dumps(response_out, indent=2)}")
                if "generated_text" not in response_out:
                    raise ValueError("Invalid response format: missing generated_text")
                return response_out["generated_text"]
                
            else:
                raise NotImplementedError(f"Model {model_name} not implemented")
                
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            logger.error(f"Response content: {response.text}")
            raise

    def guardrail_endpoint(self):
        def end_point(input: str) -> str:
            logger.debug(f"Guardrail endpoint called with input length: {len(input)}")
            logger.debug(f"Using model: {self.model}")
            logger.debug(f"Using temperature: {self.temperature}")
            logger.debug(f"Using max tokens: {self.max_tokens}")
            logger.debug(f"Using system message: {self.system_message}")

            try:
                # Convert exp version to preview version if needed
                model_name = self.model
                if "gemini-2.5-pro-exp" in model_name:
                    model_name = model_name.replace("exp", "preview")
                    logger.info(f"Converting model name from {self.model} to {model_name}")

                payload = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": self.system_message},
                        {"role": "user", "content": input},
                    ],
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                }
                if self.other_parameters:
                    payload.update(self.other_parameters)

                response = httpx.post(
                    self.end_point,
                    headers=self.headers,
                    json=payload,
                    timeout=60.0  # 60 seconds timeout
                )
                response.raise_for_status()
                return self.parse_response(response)
            except Exception as e:
                logger.error(f"Error in guardrail_endpoint: {str(e)}")
                logger.error(f"Full error details: {e}")
                raise RuntimeError(f"OpenAI API error: {str(e)}") from e

        return end_point

class Chat:
    def __init__(
        self,
        model: str,
        api_key: str,
        api_base: str,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self._session = None
        self._lock = threading.Lock()

    def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self._session

    async def _make_request(
        self, messages: List[Dict[str, str]], temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Make API request with retry mechanism."""
        try:
            # Validate inputs
            if not isinstance(messages, list):
                raise TypeError("messages must be a list")
            if not all(isinstance(msg, dict) for msg in messages):
                raise TypeError("All messages must be dictionaries")
            if not all("role" in msg and "content" in msg for msg in messages):
                raise ValueError("All messages must have 'role' and 'content' fields")
            if not isinstance(temperature, (int, float)):
                raise TypeError("temperature must be a number")
            if not 0 <= temperature <= 1:
                raise ValueError("temperature must be between 0 and 1")
            
            session = self._get_session()
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            
            data = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
            }
            
            for attempt in range(self.max_retries):
                try:
                    async with session.post(
                        f"{self.api_base}/v1/chat/completions",
                        headers=headers,
                        json=data,
                    ) as response:
                        if response.status == 200:
                            try:
                                result = await response.json()
                                if not isinstance(result, dict):
                                    raise ValueError("Invalid response format: not a dictionary")
                                if "choices" not in result:
                                    raise ValueError("Invalid response format: missing choices")
                                return result
                            except json.JSONDecodeError as e:
                                raise ValueError(f"Invalid JSON response: {str(e)}")
                            
                        elif response.status == 422:
                            error_text = await response.text()
                            if "must have less than" in error_text:
                                raise LongerThanContextError(
                                    f"Input too long for model {self.model}"
                                )
                            raise ValueError(f"Validation error: {error_text}")
                            
                        elif response.status == 429:
                            if attempt < self.max_retries - 1:
                                retry_after = response.headers.get("Retry-After", self.retry_delay)
                                await asyncio.sleep(float(retry_after) * (2 ** attempt))
                                continue
                            raise Exception("Rate limit exceeded")
                            
                        elif response.status == 401:
                            raise ValueError("Invalid API key")
                            
                        elif response.status == 403:
                            raise ValueError("Insufficient permissions")
                            
                        elif response.status >= 500:
                            if attempt < self.max_retries - 1:
                                await asyncio.sleep(self.retry_delay * (2 ** attempt))
                                continue
                            raise Exception("Server error")
                            
                        else:
                            error_text = await response.text()
                            raise Exception(f"API error: {error_text}")
                            
                except aiohttp.ClientError as e:
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay * (2 ** attempt))
                        continue
                    raise Exception(f"Network error: {str(e)}")
                
            raise Exception("Max retries exceeded")
            
        except Exception as e:
            logger.error(f"Error in _make_request: {str(e)}")
            raise

    async def cleanup(self) -> None:
        """Clean up resources."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    def __del__(self) -> None:
        """Ensure cleanup on deletion."""
        if self._session and not self._session.closed:
            asyncio.create_task(self.cleanup())