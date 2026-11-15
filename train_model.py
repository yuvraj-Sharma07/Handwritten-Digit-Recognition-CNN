"""
Group 3: Handwritten Digit Recognition with Explainable CNN
Train/evaluate on MNIST. Run: python train_model.py
"""
import os, json, time, random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

def build_model():
    inputs = tf.keras.Input(shape=(28, 28, 1), name="digit_image")
    x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu", name="conv1")(inputs)
    x = tf.keras.layers.MaxPooling2D(name="pool1")(x)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu", name="conv2")(x)
    x = tf.keras.layers.MaxPooling2D(name="pool2")(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu", name="last_conv")(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dropout(0.25, name="dropout")(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax", name="predictions")(x)
    model = tf.keras.Model(inputs, outputs, name="explainable_digit_cnn")
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def main():
    (x_train_all, y_train_all), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train_all = x_train_all.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train_all = x_train_all[..., np.newaxis]
    x_test = x_test[..., np.newaxis]

    # Fixed 5,000-image validation split from the official training partition.
    x_train, y_train = x_train_all[5000:], y_train_all[5000:]
    x_val, y_val = x_train_all[:5000], y_train_all[:5000]

    model = build_model()
    model.summary()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3,
                                         restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint("models/digit_cnn.keras",
                                           monitor="val_accuracy",
                                           save_best_only=True)
    ]
    start = time.perf_counter()
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                        epochs=12, batch_size=128, callbacks=callbacks, verbose=1)
    train_seconds = time.perf_counter() - start

    model = tf.keras.models.load_model("models/digit_cnn.keras")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    probs = model.predict(x_test, batch_size=256, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_text = classification_report(y_test, y_pred, digits=4, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=list(range(10)))
    disp.plot(cmap="Blues", values_format="d")
    plt.title("MNIST Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(history.history["accuracy"], label="train accuracy")
    plt.plot(history.history["val_accuracy"], label="validation accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
    plt.tight_layout(); plt.savefig("results/training_history.png", dpi=180); plt.close()

    # Warm-up then measure per-image model-only inference latency.
    sample = x_test[:1]
    for _ in range(10):
        model(sample, training=False)
    n = min(1000, len(x_test))
    t0 = time.perf_counter()
    for i in range(n):
        model(x_test[i:i+1], training=False)
    latency_ms = (time.perf_counter() - t0) * 1000.0 / n

    model_size_mb = os.path.getsize("models/digit_cnn.keras") / (1024 * 1024)
    metrics = {
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "training_seconds": float(train_seconds),
        "average_single_image_inference_ms_cpu_or_active_device": float(latency_ms),
        "model_file_size_mb": float(model_size_mb),
        "test_samples": int(len(y_test)),
        "note": "Record hardware, TensorFlow version, and date when reporting results."
    }
    with open("results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open("results/classification_report.txt", "w") as f:
        f.write(report_text)
    print("\n=== MEASURED TEST RESULTS ===")
    print(json.dumps(metrics, indent=2))
    print("\n=== CLASSIFICATION REPORT ===")
    print(report_text)
    print("\nSaved: models/digit_cnn.keras and results/*")

if __name__ == "__main__":
    main()
