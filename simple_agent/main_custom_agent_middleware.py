from dataclasses import dataclass
from dotenv import load_dotenv
import time

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()


class HooksDemo(AgentMiddleware):

    def __init__(self):
        super().__init__()
        self.start_time = 0.0

    def before_agent(self, state: AgentState, runtime):
        self.start_time = time.time()
        print('Before agent triggered')

    def before_model(self, state: AgentState, runtime):
        print('Before model triggered')

    def after_model(self, state: AgentState, runtime):
        print('After model triggered')

    def after_agent(self, state: AgentState, runtime):
        print('After agent finished', time.time() - self.start_time)


agent = create_agent(
    model = 'gpt-4.1-mini',
    middleware = [HooksDemo()]
)

conversation = [
        SystemMessage(content='You are helpful Assistant'),
        HumanMessage(content='Explain Principal Component Analysis in 1 line?')
    ]

response = agent.invoke(
    {
        'messages': conversation,
    }
)

print(response['messages'][-1].content)