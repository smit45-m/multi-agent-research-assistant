"""
LangChain callbacks for system observability, tracking usage, latency, and logging.
"""
import time
from typing import Dict, Any, List, Union

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from app.utils.logger import setup_logger

logger = setup_logger(__name__, "INFO")

class TokenUsageCallback(BaseCallbackHandler):
    """
    Callback handler to track token usage across LLM calls.
    Maintains a running total of prompt, completion, and total tokens.
    """
    def __init__(self):
        super().__init__()
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.successful_requests = 0

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Runs when LLM finishes running, extracts token usage from LLM output."""
        if response.llm_output and "token_usage" in response.llm_output:
            usage = response.llm_output["token_usage"]
            self.total_tokens += usage.get("total_tokens", 0)
            self.prompt_tokens += usage.get("prompt_tokens", 0)
            self.completion_tokens += usage.get("completion_tokens", 0)
            self.successful_requests += 1
            
            logger.debug(
                f"LLM Call Usage - Prompt: {usage.get('prompt_tokens', 0)}, "
                f"Completion: {usage.get('completion_tokens', 0)}, "
                f"Total: {usage.get('total_tokens', 0)}"
            )

    def get_summary(self) -> Dict[str, int]:
        """Returns a summary of token usage."""
        return {
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "successful_requests": self.successful_requests,
        }


class LatencyCallback(BaseCallbackHandler):
    """
    Callback handler to track execution time (latency) for chains and LLMs.
    """
    def __init__(self):
        super().__init__()
        self.llm_start_times: Dict[str, float] = {}
        self.chain_start_times: Dict[str, float] = {}
        self.total_llm_time = 0.0
        self.total_chain_time = 0.0

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> Any:
        """Record start time of LLM call."""
        run_id = str(kwargs.get('run_id', id(prompts)))
        self.llm_start_times[run_id] = time.time()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Calculate and record duration of LLM call."""
        run_id = str(kwargs.get('run_id'))
        if run_id in self.llm_start_times:
            duration = time.time() - self.llm_start_times[run_id]
            self.total_llm_time += duration
            logger.debug(f"LLM call {run_id} completed in {duration:.3f} seconds.")
            del self.llm_start_times[run_id]

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Record start time of Chain execution."""
        run_id = str(kwargs.get('run_id'))
        self.chain_start_times[run_id] = time.time()

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Calculate and record duration of Chain execution."""
        run_id = str(kwargs.get('run_id'))
        if run_id in self.chain_start_times:
            duration = time.time() - self.chain_start_times[run_id]
            self.total_chain_time += duration
            logger.debug(f"Chain {run_id} completed in {duration:.3f} seconds.")
            del self.chain_start_times[run_id]


class LoggingCallback(BaseCallbackHandler):
    """
    Callback handler that provides structured logging for all major LangChain events.
    Useful for debugging and auditing chain execution flows.
    """
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> Any:
        logger.info(f"Chain execution started. Inputs keys: {list(inputs.keys())}")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        logger.info(f"Chain execution finished successfully.")

    def on_chain_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> Any:
        logger.error(f"Chain execution failed with error: {str(error)}")

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> Any:
        tool_name = serialized.get("name", "Unknown Tool")
        logger.info(f"Tool '{tool_name}' started with input: {input_str[:100]}...")

    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        logger.info(f"Tool execution completed. Output length: {len(output)}")
        
    def on_tool_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> Any:
        logger.error(f"Tool execution failed with error: {str(error)}")
