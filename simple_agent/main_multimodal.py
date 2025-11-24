from dotenv import load_dotenv
from base64 import b64encode

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage

load_dotenv()

model = init_chat_model('gpt-4.1-mini')

message = HumanMessage(
    content = [
        {'type': 'text', 'text': 'Describe contents of this image'},
        {
            'type': 'image', # can pass url as well
            'base64': b64encode(open('image.png', 'rb').read()).decode(),
            'mime_type': 'image/png'
        }
    ]
)

response = model.invoke([message])

print(response.content)