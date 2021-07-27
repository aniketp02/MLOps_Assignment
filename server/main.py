import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException

# import Pycaret Regression model
import pycaret.classification as pyclf

# Import other necessary packages
from dotenv import load_dotenv
import pandas as pd
import os

# Load the environment variables from the .env file into the application
load_dotenv()

# Initialize the FastAPI application
app = FastAPI()


# Create a class to store the deployed model & use it for prediction
class Model:
    def __init__(self, modelname):

        # Load the deployed model from Amazon S3
        self.model = pyclf.load_model(modelname, platform='aws', authentication={'bucket': 'mlopsassignment20d070011'})

    def predict(self, data):

        # After predicting, we return only the column containing the predictions after converting it to a list
        predictions = pyclf.predict_model(self.model, data=data).Class.to_list()
        return predictions


# instantiating the extra trees model
model_et = Model("et_deployed")

# instantiating the random forest model
model_rf = Model("rf_deployed")

@app.get("/")
def read_root():
    return {"Hello There buddy!!"}

# Creating the POST endpoint with path '/et/predict'
@app.post("/et/predict")
async def create_upload_file(file: UploadFile = File(...)):
    # Handle the file only if it is a CSV
    if file.filename.endswith(".csv"):

        # Creating a temporary file with the same name as the uploaded CSV file
        # so that the data can be loaded into a pandas Dataframe
        with open(file.filename, "wb")as f:
            f.write(file.file.read())
        data = pd.read_csv(file.filename)
        os.remove(file.filename)

        # Return a JSON object containing the model predictions on the data
        return {
            "Labels": model_et.predict(data)
        }

    else:
        # Raise a HTTP 400 Exception, indicating Bad Request (you can learn more about HTTP response status codes here)
        raise HTTPException(status_code=400, detail="Invalid file format. Only CSV Files accepted.")


# creating POST endpoint with path '/rf/predict'
@app.post("/rf/predict")
async def create_upload_file(file: UploadFile = File(...)):
    # Handle the file only if it is a CSV
    if file.filename.endswith(".csv"):

        # Creating a temporary file with the same name as the uploaded CSV file
        # so that the data can be loaded into a pandas Dataframe
        with open(file.filename, "wb")as f:
            f.write(file.file.read())
        data = pd.read_csv(file.filename)
        os.remove(file.filename)

        # Return a JSON object containing the model predictions on the data
        return {
            "Labels": model_rf.predict(data)
        }

    else:
        # Raise a HTTP 400 Exception, indicating Bad Request (you can learn more about HTTP response status codes here)
        raise HTTPException(status_code=400, detail="Invalid file format. Only CSV Files accepted.")


# Checking if the necessary environment variables for AWS access are available. If not, exit the program
if os.getenv("AWS_ACCESS_KEY_ID") == None or os.getenv("AWS_SECRET_ACCESS_KEY") == None:
    print("AWS Credentials missing. Please set required environment variables.")
    exit(1)