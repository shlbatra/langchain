from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_classic.agents import initialize_agent, AgentType, load_tools

load_dotenv()

def generate_pet_name(animal_type, pet_color):
    llm = ChatAnthropic(model="claude-3-5-haiku-20241022", temperature=0.6)
    prompt_template_name = ChatPromptTemplate.from_template(
        "I have a {animal_type} pet with {pet_color} color, and I want a cool name for it. Suggest me five cool names for my pet"
    )

    name_chain = prompt_template_name | llm

    response = name_chain.invoke({'animal_type': animal_type, 'pet_color': pet_color})
    return response.content

def langchain_agent():
    llm = ChatOpenAI(temperature=0.5)

    tools = load_tools(["wikipedia", "llm-math"], llm = llm)

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    result = agent.invoke({
        "input": "What is the average lifespan of a dog rounded ? Multiply the age by 3"
    })

    print(result)

if __name__ == "__main__":
    langchain_agent()