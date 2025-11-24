from dataclasses import dataclass
import requests
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()


@tool('get_weather', description='Return weather information for given city', return_direct=False)
def get_weather(city: str):
    response = requests.get(f'https://wttr.in/{city}?format=j1')
    return response.json()


agent = create_agent(
    model = 'gpt-4.1-mini',
    tools = [get_weather],
)

conversation = [
        SystemMessage(content='You are helpful weather assistant, who always cracks jokes and is humorus while remaning helpful'),
        HumanMessage(content='What is weather in Toronto?')
    ]

response = agent.invoke({
    'messages': conversation
})

print(response)
print(response['messages'][-1].content)

# Stream response example - WIP on how it works

# for chunk in agent.stream({'messages': conversation}):
#     if 'messages' in chunk and chunk['messages']:
#         print(chunk['messages'][-1].content, end='', flush=True)