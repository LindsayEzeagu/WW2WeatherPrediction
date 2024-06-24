import pandas as pd
import numpy as nd

dataset = pd.read_csv('Summary of Weather.csv')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


plt.scatter(dataset['MaxTemp'], dataset['MinTemp'])
plt.title('Maximum Temperature vs Minimum Temperature')
plt.xlabel('Maximum Temperature')
plt.ylabel('Minimum Temperature')
plt.show()

X = dataset['MaxTemp'].values.reshape(-1,1)
y = dataset['MinTemp'].values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y)

X_train

y_Prediction = model.predict(X_test)

y_test

model = LinearRegression().fit(X_train, y_train)

model_accuracy = model.score(X_test, y_test)

print(model_accuracy)

plt.scatter(X_test, y_test)
plt.plot(X_test, y_Prediction, color='r')
plt.title('Maximum Temperature vs Minimum Temperature')
plt.xlabel('Maximum Temperature')
plt.ylabel('Minimum Temperature')
plt.show()
