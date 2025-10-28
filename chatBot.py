from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from dotenv import load_dotenv

load_dotenv()

LLm = HuggingFaceEndpoint(
    repo_id='mistralai/Mistral-7B-Instruct-v0.2',
    task="text-generation"
)

model = ChatHuggingFace(llm=LLm)
chat_history = [SystemMessage(content="You are an Helpful Agent.")]

while True :
    user_input = input("Ask Anything... ").upper()
    chat_history.append(HumanMessage(content=user_input))
    if user_input == 'EXIT' :
        break
    result = model.invoke(user_input)
    chat_history.append(AIMessage(content=result.content))
    print('AI :',result.content)

print(chat_history)