from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage 
'''(There is No Necessary of this above import)'''

# chat_Template = ChatPromptTemplate(
#     [('system','Your are a helpful {Domain} expert'),
#      ('human','Tell me about this {Topic}')]
# )

# Actually it should be used with System Message and Human Message, but it wont work 
# Code Snippet for Reference

chat_Template = ChatPromptTemplate([
    SystemMessage(content='Your an Helpful {Domain} Expert'),
    HumanMessage(content='Tell Me about this {Topic}')
    ])

prompt = chat_Template.invoke({
    'Domain':'AI',
    'Topic':'Explain me the Transformer Model.'
})

print(prompt)