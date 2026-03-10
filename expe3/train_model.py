import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("train.csv")

# Select useful features
df = df[['Pclass','Sex','Age','SibSp','Parch','Fare','Survived']]

# Convert Sex to numeric
df['Sex'] = df['Sex'].map({'male':0,'female':1})

# Fill missing Age
df['Age'] = df['Age'].fillna(df['Age'].mean())

X = df.drop('Survived',axis=1)
y = df['Survived']

model = LogisticRegression(max_iter=1000)
model.fit(X,y)

# Save model
pickle.dump(model,open("model.pkl","wb"))

print("Model saved successfully")