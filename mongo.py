import pymongo
import os
from dotenv import load_dotenv
load_dotenv()

mongo_connection_string=os.getenv('MONGO_DB')
def readmango(team1,team2):

    client=pymongo.MongoClient(mongo_connection_string)
    db=client.Ml_project
    match=db.matchsummary
    databases = client.list_database_names()
    print("Databases available:", databases)

    # Check if the target database is listed
    if 'ML_project' in databases:
        print("Successfully connected to the database 'ML_project'.")
    else:
        print("Database 'ML_project' does not exist.")
    collection_name='matchsummary'
    if collection_name in db.list_collection_names():
        query = {
    "$or": [
        {"team1": team1, "team2": team2},
        {"team1": team2, "team2": team1}
    ]
}


        document = match.find_one(query)
        if document:
            print(document)
    
    else:
        print("No Colelction found")
   

# if __name__=="__main__":
#     input_data = {
#     "batting_team": "Chennai Super Kings",
#     "bowling_team": "Mumbai Indians",
#     "runs_scored": 150,
#     "wickets_remaining": 3,
#     "overs_remaining": 5,
#     "current_run_rate": 8.5,
#     "required_run_rate": 10.2,
#     "Winner_prediction":"Mumbai Indians"
# }
#     external_data=readmango(input_data.get('batting_team'),input_data.get('bowling_team'))