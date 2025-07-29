import os
from typing import Union, List, Dict, Optional, Any
import json
import requests

class Message:
    """Simple message class to mimic OpenAI's message format"""
    def __init__(self, content):
        self.content = content
        self.model = ""
        self.created = 0
        self.choices = []

class GLMModel:
    """GLM-4.5 API Model interface for smolagents CodeAgent"""
    
    def __init__(self, 
                 model_id='glm-4-plus',
                 max_tokens=2096,
                 temperature=0.5,
                 custom_role_conversions=None):
        """Initialize the GLM-4.5 API Model.
        
        Args:
            model_id: The model ID for GLM (e.g., 'glm-4-plus')
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            custom_role_conversions: Custom role mappings if needed
        """
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.custom_role_conversions = custom_role_conversions or {}
        
        # GLM API configuration
        self.api_key = os.environ.get("GLM_API_KEY")
        self.base_url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
        
        if not self.api_key:
            raise ValueError("GLM_API_KEY environment variable is required")
    
    def __call__(self, prompt: Union[str, dict, List[Dict]]) -> Message:
        """Make the class callable as required by smolagents"""
        try:
            # Handle different prompt formats
            if isinstance(prompt, (dict, list)):
                if isinstance(prompt, list) and all(isinstance(msg, dict) for msg in prompt):
                    messages = prompt
                    return self._generate_chat_response_message(messages)
                else:
                    prompt_str = str(prompt)
                    return self._generate_text_response_message(prompt_str)
            else:
                prompt_str = str(prompt)
                return self._generate_text_response_message(prompt_str)
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            print(error_msg)
            return Message(error_msg)
    
    def generate(self, 
                 prompt: Union[str, dict, List[Dict]],
                 stop_sequences: Optional[List[str]] = None,
                 seed: Optional[int] = None,
                 max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None,
                 **kwargs) -> Message:
        """
        Generate a response from the model.
        This method is required by smolagents and provides a more complete interface
        with support for all parameters needed by smolagents.
        
        Args:
            prompt: The prompt to send to the model.
            stop_sequences: List of sequences where the model should stop generating
            seed: Random seed for reproducibility
            max_tokens: Maximum tokens to generate (overrides instance value if provided)
            temperature: Sampling temperature (overrides instance value if provided)
            **kwargs: Additional parameters
                
        Returns:
            Message: A Message object with the response content
        """
        current_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        current_temperature = temperature if temperature is not None else self.temperature
            
        try:
            if isinstance(prompt, (dict, list)):
                if isinstance(prompt, list) and all(isinstance(msg, dict) for msg in prompt):
                    messages = prompt
                    result = self._generate_chat_response_message(messages, stop_sequences, current_max_tokens, current_temperature)
                    return result
                else:
                    prompt_str = str(prompt)
                    result = self._generate_text_response_message(prompt_str, stop_sequences, current_max_tokens, current_temperature)
                    return result
            else:
                prompt_str = str(prompt)
                result = self._generate_text_response_message(prompt_str, stop_sequences, current_max_tokens, current_temperature)
                return result
                
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            print(error_msg)
            return Message(error_msg)
    
    def _make_api_request(self, messages: List[Dict], stop_sequences: Optional[List[str]] = None, max_tokens: int = None, temperature: float = None) -> Dict:
        """Make API request to GLM-4.5"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature or self.temperature,
        }
        
        if stop_sequences:
            payload["stop"] = stop_sequences
        
        response = requests.post(self.base_url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    
    def _generate_chat_response(self, messages: List[Dict], stop_sequences: Optional[List[str]] = None, max_tokens: int = None, temperature: float = None) -> str:
        """Generate a response from the chat API and return string content"""
        try:
            response_data = self._make_api_request(messages, stop_sequences, max_tokens, temperature)
            
            if 'choices' in response_data and len(response_data['choices']) > 0:
                message = response_data['choices'][0]['message']
                
                # Check if the model wants to call a tool (GLM-4.5 tool calling format)
                if 'tool_calls' in message and message['tool_calls']:
                    code_lines = []
                    # Handle multiple tool calls if the model requests them
                    for tool_call in message['tool_calls']:
                        function_name = tool_call['function']['name']
                        try:
                            function_args = json.loads(tool_call['function']['arguments'])
                            args_list = [f"{key}={repr(value)}" for key, value in function_args.items()]
                            args_str = ", ".join(args_list)
                            
                            # Assign the result to a unique variable to avoid conflicts
                            result_var = f"{function_name}_result_{tool_call['id'].replace('-', '_')}"
                            code_lines.append(f"{result_var} = {function_name}({args_str})")

                        except json.JSONDecodeError:
                             # If args fail to parse, still attempt to call the function if no args are needed.
                            code_lines.append(f"{function_name}()")

                    # Crucially, call final_answer with the result of the *last* tool call.
                    # This terminates the smolagents loop correctly.
                    if code_lines:
                        last_result_var = code_lines[-1].split(' = ')[0]
                        code_lines.append(f"final_answer({last_result_var})")

                    code_to_execute = "\n".join(code_lines)
                    
                    thought = "I will execute the requested tool(s) and provide the final result."
                    smol_response = f"""Thought: {thought}
Code:
```py
{code_to_execute}
```<end_code>"""
                    return smol_response

                # If no tool_calls, return the content as is.
                return message.get('content', '') or ""
            else:
                return "No response generated"
                
        except Exception as e:
            return f"Error in API request: {str(e)}"
    
    def _generate_chat_response_message(self, messages: List[Dict], stop_sequences: Optional[List[str]] = None, max_tokens: int = None, temperature: float = None) -> Message:
        """Generate a response from the chat API and return a Message object"""
        content = self._generate_chat_response(messages, stop_sequences, max_tokens, temperature)
        return Message(content)
    
    def _generate_text_response(self, prompt: str, stop_sequences: Optional[List[str]] = None, max_tokens: int = None, temperature: float = None) -> str:
        """Generate a response from the text completion API and return string content"""
        messages = [{"role": "user", "content": prompt}]
        return self._generate_chat_response(messages, stop_sequences, max_tokens, temperature)
        
    def _generate_text_response_message(self, prompt: str, stop_sequences: Optional[List[str]] = None, max_tokens: int = None, temperature: float = None) -> Message:
        """Generate a response from the text completion API and return a Message object"""
        content = self._generate_text_response(prompt, stop_sequences, max_tokens, temperature)
        return Message(content)