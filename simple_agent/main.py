from dataclasses import dataclass
import requests
from dotenv import load_dotenv
import os

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

# Context of user_id 
@dataclass
class Context:
    user_id: str

# Response format
@dataclass
class ResponseFormat:
    summary: str
    temperature_celsius: float
    temperature_fahrenheit: float
    humidity: float

@tool('get_weather', description='Return weather information for given city', return_direct=False)
def get_weather(city: str):
    response = requests.get( f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={city}", timeout=15)
    return response.json()

@tool('locate_user', description="Look up user city based on context")
def locate_user(runtime: ToolRuntime[Context]): 
    match runtime.context.user_id: # shared memory space that tools can access during execution (shared workspace that persists during the conversation)
        case 'ABC123':
            return 'Vienna'
        case 'XYZ456':
            return 'London'
        case 'HJKL111':
            return 'Paris'
        case _:
            return 'Unknown'

model = init_chat_model('gpt-4.1-mini', temperature=0.5)

checkpointer = InMemorySaver()

agent = create_agent(
    model = model,
    tools = [get_weather, locate_user],
    system_prompt = 'You are helpful weather assistant, who always cracks jokes and is humorus while remaning helpful',
    context_schema = Context, # contains user id
    response_format=ResponseFormat, # model respond in specific format
    checkpointer=checkpointer # add memory to model based on thread_id
)

conversation = [
        # SystemMessage(content='You are helpful weather assistant, who always cracks jokes and is humorus while remaning helpful'),
        HumanMessage(content='How is weather in Toronto ?')
    ]

configs = {'configurable':{'thread_id':1}}

response = agent.invoke({
    'messages': conversation
    },
    config=configs,
    context=Context(user_id='ABC123')
)

#print(response)
#print(response['structured_response'])
print(response['structured_response'].summary)
print(response['structured_response'].temperature_celsius)


# Continue conversation

configs = {'configurable':{'thread_id':2}} # change thread to test if conversation breaks 
# It's a sunny day in Toronto with a bit of a cool breeze from the south-southwest. The temperature is mild, perfect for a walk if you don't mind a little chill!
# I need to know the specific weather condition or location you're referring to in order to determine if it's usual or not. Could you please provide more details or specify the city or weather condition?


conversation2 = [
        HumanMessage(content='And is this usual ?')
    ]

response = agent.invoke({
    'messages': conversation2
    },
    config=configs,
    context=Context(user_id='ABC123')
)

#print(response)
#print(response['structured_response'])
print(response['structured_response'].summary)
print(response['structured_response'].temperature_celsius)


# Stream response example - WIP on how it works

# for chunk in agent.stream({'messages': conversation}):
#     if 'messages' in chunk and chunk['messages']:
#         print(chunk['messages'][-1].content, end='', flush=True)