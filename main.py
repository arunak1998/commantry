import pymongo
from mongo import readmango
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from langsmith import traceable
from dotenv import load_dotenv
load_dotenv()

groq_api_key=os.getenv('GROQ_API_KEY')

def get_model():
    model=ChatGroq(api_key=groq_api_key,model='llama3-70b-8192')
    return model

def getprompt():
    prompt_template = PromptTemplate.from_template(
  """You are an Cricket Commentator who produces Excellent Real time Commantry based on the Provided scores of the teams Chasing  You have to be more Interactive and Produce Awesome Commantry 
  You will be Provided with Run scored Remaining Overs Remaining Wickets Required Run rate and Current Run Rate with Prediction result from the ML model so You have to give the Interactive Commantry and How Much Have to go
  and Some scenarious to make more Ineractive Don not Provide Aany Wrong Infromation in Commantry You have to be More Presise and more fun and serious way of Commantry
 
 
 "Key Points"
  if the Overs are less and required run Rate is Higheer Give the scenario of Bowlers and batsman and Match heat
  If the Runs is Easliy gettable and Batting team scoring Faster give some Inovatiove comments on more Enjoayable
  If the Early wickts fall get some more attention in commantry to give some Senarios and Knwoledgable to gaming 
  
  Always Keep Eye on the Prediction winner and Produce Commantry 
  always take COnsier all parameters and also take Consider of External Proameter which only provided 
  Dont Provide any Unnecessary Player deatils and Stats You don't have Focus Only on the Provided Data  and External data

  input_data:{input}
  external_data:{external}
  """
)


   
    return prompt_template
@traceable
def createchain():

    chain=getprompt() | get_model() | StrOutputParser()

    input_data = {
    "batting_team": "Chennai Super Kings",
    "bowling_team": "Kolkata Knight Riders",
    "runs_scored": 110,
    "wickets_remaining": 5,
    "overs_remaining": 8,
    "current_run_rate": 9.5,
    "required_run_rate": 9.2,
    "Winner_prediction":"Kolkata Knight Riders"
}
    external_data=readmango(input_data.get('batting_team'),input_data.get('bowling_team'))


    print(chain.invoke({"input":input_data,"external":external_data}))
    

if __name__=='__main__':
    createchain()

    