from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

LLm = HuggingFaceEndpoint(
    repo_id='mistralai/Mistral-7B-Instruct-v0.2',
    task="conversational"
)

model = ChatHuggingFace(llm=LLm)
chat_history = []

while True :
    user_input = input("Ask Anything...").upper()
    chat_history.append(user_input)
    if user_input == 'EXIT' :
        break
    result = model.invoke(user_input)
    chat_history.append(result.content)
    print(result.content)

print(chat_history)