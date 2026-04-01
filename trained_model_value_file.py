#it shows which model was finally saved in the model.pkl file after training and evaluating 
# the model. It confirms that the model is of type XGBClassifier, which is expected based on 
# the training code.


import joblib
model = joblib.load("model.pkl")
print(type(model))