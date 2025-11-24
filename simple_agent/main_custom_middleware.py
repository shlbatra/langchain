from dataclasses import dataclass
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, dynamic_prompt
from langchain.messages import HumanMessage

load_dotenv()

@dataclass
class Context:
    user_role: str

# Prompt based on user role
@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    user_role = request.runtime.context.user_role

    base_prompt = 'You are helpful and very concise assistant'

    match user_role:
        case 'expert':
            return f'{base_prompt} Provide detail technical responses'
        case 'beginner':
            return f'{base_prompt} Keep your explanations simple and basic'
        case 'child':
            return f'{base_prompt} Explain everything as is you were literally talking to a five year old'
        case _:
            return base_prompt
        
agent = create_agent(
    model = 'gpt-4.1-mini',
    middleware = [user_role_prompt],
    context_schema = Context
)

conversation = [
        HumanMessage(content='Explain Principal Component Analysis ?')
    ]

response = agent.invoke(
    {
        'messages': conversation,
    },
    context = Context(user_role='child')
)

print(response['messages'][-1].content)