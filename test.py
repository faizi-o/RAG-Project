import sys
from dotenv import load_dotenv
load_dotenv()

import langchain
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

model = ChatMistralAI(model ="mistral-small-2603")


meassages=[ 
   SystemMessage(content="You are a helpful assistant that solve user's problem and answer the question in a friendly manner."),

]
print ("---------------i am faiz's pesonal chat bot-------------------------- ")
while True:
    prompt = input('YOU : ')
    if prompt == "0" :
     break
    meassages.append(HumanMessage(content=prompt))  
    respone = model.invoke(meassages)
    meassages.append(AIMessage(content=respone.content))
    print("BOT :" , respone.content)

print (meassages)    

