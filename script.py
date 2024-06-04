
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, prediction_score
import sklearn
import joblib
import boto3
import pathlib
from io import StringIO
import argparse
import os
import numpy as np
import pandas as pd



def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))  #the function loads the model from the model_dir
    return clf
    
if __name__ == "__main__": #execution starts from here, amd these arguments are required by default
    print("[INFO] Extrating arguments")
    parser = argparse.ArgumentParser()
    
    #hyperparameters sent by the client are passed as command line arguments 
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--random_state", type=int, default=0)
    
    #Data, model, and output directories
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument("--output_dir", type=str, default="test.1.csv")
    parser.add_argument("--output_dir", type=str, default="train.1.csv")
    
    args, _ = parser.parse_known_args()
    
    
    print("SkLearn version:".sklearn.__version__)
    print("SKLearn version:".joblib.__version__)
    
    print("[INFO] Reading data")
    print()
    
    #load the data from the s3 bucket
    train_df = pd.read_csv(os.path.join(args.train, args.train_file ))
    test_df = pd.read_csv(os.path.join(args.test, args.test_file))
    
    features  = list(train_df.columns)
    label = features.pop(-1)
    
    print("Building training and testing datasets")
    print()
    X_train = train_df[features]
    X_test = test_df[features]
    y_train = train_df[label]
    y_test = test_df[label]
    
    print("Column order:")
    print(features)
    print()
    
    print("label column is : ", label)
    print()
    
    print("Data Shape: ")
    print()
    
    print("---- SHAPE OF TRAINING DATA (85%) ----")
    print(X_train.shape)
    print(y_train.shape)    
    print()
    print("---- SHAPE OF TESTING DATA (15%) ----")
    print(X_test.shape)
    print(y_test.shape)
    print()
    
    print("Training RandomForest Model....")
    print()
    model = RandomForestClassifier(n_estimators=args.n_estimators, random_state=args.random_state,vars=args.vars)
    model.fit(X_train, y_train)
    print()
    
    model_path = os.path.join(args.model_dir, "model.job") 
    joblib.dump(model, model_path)
    print("Model persisted at + model_path")
    print()
    
    
    y_pred = model.predict(X_test)    
    test_acc = accuracy_score(y_test, y_pred)
    test_rep = classification_report(y_test, y_pred)
    
    print()
    print("----MERTICS RESULTS FOR TESTING DATA----")
    print()
    print("Total Rows Are: ", X_test.shape[0])
    print("[TESTING] Model Accuracy: ", test_acc)
    print("[TESTING] Testing Report: ")
    print(test_rep)    
    
    
