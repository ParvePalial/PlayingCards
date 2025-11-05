import joblib
import cv2
from skimage.feature import hog

clf, encoder = joblib.load("card_classifier.joblib")

IMG_SIZE = (200, 200)

def predict_card(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, IMG_SIZE)
    features = hog(img, orientations=9, pixels_per_cell=(16,16),
                   cells_per_block=(2,2), block_norm='L2-Hys')
    pred = clf.predict([features])[0]
    return encoder.inverse_transform([pred])[0]


for i in range(1,6):
    print(predict_card(f"/Users/papalial/Desktop/Code/AI/Datasets/PlayingCardsClassification/test/joker/{i}.jpg"))
