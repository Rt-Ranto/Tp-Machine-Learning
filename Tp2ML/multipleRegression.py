from pathlib import Path
import pandas as pd
from sklearn import linear_model


path = Path(__file__).resolve().parent / "Dirtydata.csv"
df = pd.read_csv(path)
#donnée sans NaN
train = df[df['Calories'].notna()]
#Seulement les donnée avec NaN
test = df[df['Calories'].isna()]

X_train = train[['Duration','Pulse','Maxpulse']]
y_Train = train['Calories']

X_test = test[['Duration','Pulse','Maxpulse']]

regr = linear_model.LinearRegression()
regr.fit(X_train, y_Train)

df.loc[df['Calories'].isna(),'Calories'] = regr.predict(X_test)

a = regr.predict([[60,92,115]])
print(df.to_string())
print(regr.coef_)
print(a)
