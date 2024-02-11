# Import libraries
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Load the global model
loaded_model = tf.keras.models.load_model('global-model.tf')

# Make predictions
y_pred = loaded_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)


# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred_classes)

# Calculate precision, recall, and F1-score
precision = precision_score(y_test, y_pred_classes, average='weighted')
recall = recall_score(y_test, y_pred_classes, average='weighted')
f1 = f1_score(y_test, y_pred_classes, average='weighted')

# Create a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_classes)

# Display and plot the results
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-Score: {f1}')
print('Confusion Matrix:')
print(conf_matrix)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(10)
plt.xticks(tick_marks, range(10), rotation=45)
plt.yticks(tick_marks, range(10))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()