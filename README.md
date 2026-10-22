# ai-based-diabetes-prediction
AI-Based Diabetes Prediction System using Machine Learning and Streamlit

The project is an AI-Based Diabetes Prediction System developed using Python and machine learning. First, I prepared a diabetes health-indicator dataset and separated the input features from the target column, Diabetes_binary. I divided the data into training and testing sets and trained a Random Forest classifier. After evaluating the model, I saved the compact model using Joblib.
I then developed a web application using Streamlit. The website collects 21 health-related inputs from the user and converts them into a Pandas DataFrame in the feature order used during training. The saved model generates a class-1 score, and the application compares that score with a threshold of 0.30. It displays either a positive or negative classification. This is an academic prototype and not a medical diagnosis system.
