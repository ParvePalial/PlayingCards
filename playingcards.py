import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ---------------- CONFIG ----------------
DATASET_DIR = "/Users/papalial/Desktop/Code/AI/Datasets/PlayingCardsClassification/train"  # 👈 change this
IMG_SIZE = (200, 200)  # resize to match your dataset

# ---------------- LOAD DATA ----------------
X, y = [], []

for label in sorted(os.listdir(DATASET_DIR)):
    class_dir = os.path.join(DATASET_DIR, label)
    if not os.path.isdir(class_dir):
        continue

    print(f"Loading {label} ...")
    for img_name in os.listdir(class_dir):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(class_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img = cv2.resize(img, IMG_SIZE)

        # Extract HOG features (robust for card edges/shapes)
        features = hog(
            img,
            orientations=9,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            block_norm='L2-Hys'
        )

        X.append(features)
        y.append(label)

X = np.array(X)
y = np.array(y)
print(f"\n✅ Loaded {len(X)} images across {len(np.unique(y))} classes")


# ---------------- ENCODE LABELS ----------------
encoder = LabelEncoder()
y_enc = encoder.fit_transform(y)

# ---------------- TRAIN / TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
)

# ---------------- TRAIN CLASSIFIER ----------------
print("\nTraining SVM model...")
clf = SVC(kernel='linear', probability=True)
clf.fit(X_train, y_train)

# ---------------- EVALUATE ----------------
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n🎯 Accuracy: {acc:.3f}")
print("\n" + classification_report(y_test, y_pred, target_names=encoder.classes_))

# ---------------- SAVE MODEL ----------------
joblib.dump((clf, encoder), "card_classifier.joblib")
print("\n💾 Model saved to card_classifier.joblib")

# ---------------- TEST PREDICTION ----------------
def predict_card(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, IMG_SIZE)
    features = hog(
        img,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )
    pred = clf.predict([features])[0]
    return encoder.inverse_transform([pred])[0]

# Example usage
test_image = "/Users/papalial/Desktop/Code/AI/Datasets/PlayingCardsClassification/test/four of diamonds/1.jpg"
print("Prediction:", predict_card(test_image))
