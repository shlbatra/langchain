from dataclasses import dataclass
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain.messages import HumanMessage, SystemMessage

load_dotenv()


basic_model = init_chat_model(model='gpt-4o-mini')
advanced_model = init_chat_model(model='gpt-4.1-mini')


# Model decision here
@wrap_model_call
def dyanmic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    message_count = len(request.state['messages'])

    if message_count > 3:
        model = advanced_model
    else:
        model = basic_model
        
    request.model = model

    return handler(request)

agent = create_agent(
    model = basic_model,
    middleware = [dyanmic_model_selection]
)

conversation = [
        SystemMessage(content='You are helpful Assistant'),
        HumanMessage(content='Explain Principal Component Analysis in 1 line?'),
        HumanMessage(content='Explain Principal Component Analysis in 1 line?'),
        HumanMessage(content='Explain Principal Component Analysis in 1 line?')
    ]

response = agent.invoke(
    {
        'messages': conversation,
    }
)

print(response['messages'][-1].content)
print(response['messages'][-1].response_metadata['model_name'])