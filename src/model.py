from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, LSTM, Bidirectional, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2

def build_bilstm(input_dim):
    model = Sequential([
        Input(shape=(1, input_dim)),
        # Units dropped to 8 to prevent memorization
        # Heavy L2 on weights and activity
        Bidirectional(LSTM(8, kernel_regularizer=l2(0.08), recurrent_regularizer=l2(0.08))), 
        BatchNormalization(),
        Dropout(0.7), # Extreme dropout
        Dense(4, activation='relu', kernel_regularizer=l2(0.08)),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model