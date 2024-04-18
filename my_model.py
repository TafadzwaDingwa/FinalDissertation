import numpy as np
import pandas as pd
df = pd.read_csv("/app/data/Benign_vs_DDoS.csv")

y = df['Label']
from sklearn.preprocessing import LabelEncoder


le = LabelEncoder()
y = le.fit_transform(y)
for label, encoded_label in zip(le.classes_, le.transform(le.classes_)):
    print(f"{label} -> {encoded_label}")


class_names = list(le.classes_)
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()



import tensorflow as tf



from tensorflow.keras import layers, models


from tensorflow import keras
from tensorflow.keras import layers, models


X = df[['Min Packet Length', 'URG Flag Count', 'SYN Flag Count', 'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Total Backward Packets', 'Average Packet Size', 'ACK Flag Count', 'Inbound',  'Init Win bytes forward']]
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
X_train_scaled = scaler.fit_transform(X_train)
    
X_test_scaled = scaler.transform(X_test)
    

    
X = X.values.reshape(-1, X.shape[1], 1)

model = keras.models.Sequential([
    keras.layers.Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(10, 1), padding = 'same'),
    
    keras.layers.Dropout(0.6),
    keras.layers.Conv1D(filters=32, kernel_size=2, activation='relu'),
    
    keras.layers.Dropout(0.5),

     keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu'),
    
    keras.layers.Dropout(0.4),
    keras.layers.GlobalMaxPooling1D(),

    keras.layers.Dense(64, activation='relu'),
    
    keras.layers.Dense(1, activation='sigmoid')
])

    
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
    
model.fit(X_train_scaled, y_train, epochs=10, batch_size = 200, verbose=2)
    

test_eval = model.evaluate(X_test_scaled, y_test, verbose=2)
 
print('Test loss:', test_eval[0])
    

print('Test accuracy:', test_eval[1])
    


from sklearn.metrics import confusion_matrix, classification_report

from sklearn.metrics import precision_recall_fscore_support

y_pred = model.predict(X_test_scaled,  batch_size=128, verbose=2)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred = (y_pred > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

cm = confusion_matrix(y_test, y_pred)
print(cm)

print(classification_report(y_test, y_pred, target_names=class_names, digits = 4))
             


