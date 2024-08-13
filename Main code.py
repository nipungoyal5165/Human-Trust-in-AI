# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv(r'C:\Users\user1\Desktop\CS205.2\Trust.csv')

# Perform one-hot encoding for the 'Gender' column to convert categorical values into binary columns
data = pd.get_dummies(data, columns=['Gender'], drop_first=True)

# Select features and target variable
# Ensure the correct one-hot encoded gender column name is used
gender_columns = [col for col in data.columns if col.startswith('Gender_')]
features = data[['Year', 'Age'] + gender_columns]  # Adjust column names as needed
target = data['Trust_Level']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Print model coefficients
print('Model Coefficients:')
for feature, coef in zip(features.columns, model.coef_):
    print(f'{feature}: {coef}')

# Predict future trust levels for the next 10 years
current_year = data['Year'].max()
future_years = np.arange(current_year + 1, current_year + 11)

# Create a DataFrame for future predictions
future_data = pd.DataFrame({
    'Year': future_years,
    'Age': np.mean(data['Age']),  # Using the mean of current Age
    gender_columns[0]: 1  # Assuming the gender for prediction; adjust as necessary
})

# Make future predictions
future_predictions = model.predict(future_data)

# Combine future years with their predicted trust levels
future_data['Predicted_TrustLevel'] = future_predictions

# Print the future predictions
print(future_data)

# Plotting the results
plt.scatter(data['Year'], data['Trust_Level'], color='blue', label='Actual Trust Level')
plt.scatter(X_test['Year'], y_pred, color='red', label='Predicted Trust Level')
plt.plot(future_data['Year'], future_data['Predicted_TrustLevel'], color='green', label='Predicted Future Trust Level')
plt.xlabel('Year')
plt.ylabel('Trust Level')
plt.title('Actual and Predicted Trust Level Over Years')
plt.legend()
plt.show()
