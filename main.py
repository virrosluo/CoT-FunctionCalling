from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt

from tenacity import retry, wait_random_exponential, stop_after_attempt

from utils import *
from env import *

import json
import importlib.util

@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(messages, tools=None, tool_choice=None, model=GPT_MODEL):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e
    
    
client = OpenAI(api_key=API_KEY)

with open(FUNCTION_META, 'r') as file:
    tools = json.load(file)

# Load the module dynamically
spec = importlib.util.spec_from_file_location("computeFunc", FUNCTION_FILE)
computeFunc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(computeFunc)

functions = {
    tool["function"]["name"]: getattr(computeFunc, tool["function"]["name"])
    for tool in tools if tool["type"] == "function"
}

messages = []
messages.append({"role": "system", "content": "Each time output only 1 function call. Ask for clarification if a user request is ambiguous."})
messages.append({"role":"user", "content":"What the output of plus between ten and twenty and with another calculation is 20 divide by 2"})
while True:    
    chat_response = chat_completion_request(
        messages, 
        tools=tools, 
        tool_choice="auto")

    assistant_message = chat_response.choices[0].message
    if assistant_message.tool_calls:
        messages += ([{'role': assistant_message.role,
                    'function_call': response.function} 
                    for response in assistant_message.tool_calls])
        
        # Calling the function
        for responseMessage in assistant_message.tool_calls:
            func_name, func_params = responseMessage.function.name, json.loads(responseMessage.function.arguments)

            result = execute_function_call({
                'name': func_name,
                'args': func_params
            }, functions)

            messages.append({
                "role": "function",
                "tool_call_id": responseMessage.id,
                "name": func_name,
                'content': json.dumps(result)
            })
            
    else:
        messages += [{'role': assistant_message.role,
                    'content': assistant_message.content}]
        
        break

pretty_print_conversation(messages)