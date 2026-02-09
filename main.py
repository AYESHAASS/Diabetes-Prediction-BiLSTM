from src.preprocessing import clean_and_split
from src.model import build_bilstm
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, accuracy_score
import joblib

DATA_PATH = "data/diabetes.csv"

def run_pipeline():
    X_train, X_test, y_train, y_test, scaler = clean_and_split(DATA_PATH)

    model = build_bilstm(X_train.shape[2])

    # Aggressive Stop: If val_loss doesn't improve for 5 epochs, STOP.
    # This prevents the model from spending extra time overfitting.
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)

    print("Starting training with aggressive regularization...")
    model.fit(
        X_train, y_train, 
        epochs=100, # Can be high because EarlyStopping will kill it early
        batch_size=16, # Smaller batch size for better generalization
        validation_split=0.2, 
        verbose=1, 
        callbacks=[early_stop, reduce_lr]
    )

    # Final Metrics
    y_train_pred = (model.predict(X_train) > 0.5).astype(int)
    y_test_pred = (model.predict(X_test) > 0.5).astype(int)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    print(f"\nFinal Train Accuracy: {train_acc*100:.2f}%")
    print(f"Final Test Accuracy: {test_acc*100:.2f}%")
    print(f"Overfitting Gap: {(train_acc - test_acc)*100:.2f}%")
    
    print("\nTest Classification Report:")
    print(classification_report(y_test, y_test_pred))

if __name__ == "__main__":
    run_pipeline()