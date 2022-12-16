import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load the data into a pandas DataFrame
df = pd.read_csv('data.csv')

# Calculate the bid and ask volumes
df['bid_volume'] = df['bid'].rolling(5).sum()
df['ask_volume'] = df['ask'].rolling(5).sum()

# Calculate the delta indicator
df['delta'] = df['close'].diff(5)

# Set the threshold for the order imbalance
threshold = 1.5

# Initialize a list to store the trade signals
trade_signals = []

# Iterate through the rows of the DataFrame
for index, row in df.iterrows():
  # If the ratio of bid volume to ask volume exceeds the threshold, and there is a positive delta, buy
  if row['bid_volume'] / row['ask_volume'] > threshold and row['delta'] > 0:
    trade_signals.append(1)
  # If the ratio of ask volume to bid volume exceeds the threshold, and there is a negative delta, sell
  elif row['ask_volume'] / row['bid_volume'] > threshold and row['delta'] < 0:
    trade_signals.append(-1)
  # Otherwise, do not trade
  else:
    trade_signals.append(0)

# Add the trade signals to the DataFrame
df['trade_signals'] = trade_signals

# Split the data into features and labels
X = df[['bid_volume', 'ask_volume', 'delta']]
y = df['trade_signals']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the random forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the classifier to the training data
clf.fit(X_train, y_train)

# Make predictions on the testing data
predictions = clf.predict(X_test)

# Calculate the accuracy of the predictions
accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)
