import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Path to the trained images and test images
image_paths = [r'C:\Users\JasonHinch\PycharmProjects\FacialRecognition2\Nicolas_Cage.webp', r'C:\Users\JasonHinch\PycharmProjects\FacialRecognition2\nicolas-cage-test.png']  # Example paths
test_image_path = r'C:\Users\JasonHinch\PycharmProjects\FacialRecognition2\the-15-best-nicolas-cage-movies_bvu1.1280.webp'
output_image_path = 'Nic.jpg'

# Load the training images and prepare the training data
training_data = []
labels = []
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

for image_path in image_paths:
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (100, 100))
        training_data.append(face_img.flatten())
        labels.append("Nicholas Cage")

# Train the KNN model
model = KNeighborsClassifier(n_neighbors=1)
model.fit(training_data, labels)

# Load and process the test image
test_image = cv2.imread(test_image_path)
test_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(test_gray, 1.3, 5)

# Detect, identify and label the face in the test image
for (x, y, w, h) in faces:
    face_img = test_gray[y:y+h, x:x+w]
    face_img = cv2.resize(face_img, (100, 100))
    prediction = model.predict([face_img.flatten()])
    cv2.rectangle(test_image, (x, y), (x+w, y+h), (255, 0, 0), 3)
    cv2.putText(test_image, prediction[0], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

# Save the result
cv2.imwrite(output_image_path, test_image)

# Optionally display the image
cv2.imshow('Recognize Face', test_image)
cv2.waitKey(0)
cv2.destroyAllWindows()




# import cv2
# import numpy as np
# import os
# from sklearn.neighbors import KNeighborsClassifier
#
# # Load images
# image_paths = [r'C:\Users\JasonHinch\PycharmProjects\FacialRecognition2\Nicolas_Cage.webp', r'C:\Users\JasonHinch\PycharmProjects\FacialRecognition2\the-15-best-nicolas-cage-movies_bvu1.1280.webp']  # Add paths to Nicholas Cage images
# training_data = []
# labels = []
#
# # Haar Cascade Classifier
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#
# # Process each image
# for image_path in image_paths:
#     img = cv2.imread(image_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#     for (x, y, w, h) in faces:
#         face_img = gray[y:y+h, x:x+w]
#         face_img = cv2.resize(face_img, (100, 100))
#         training_data.append(face_img.flatten())
#         labels.append("Nicholas Cage")
#
# # Train KNN classifier
# model = KNeighborsClassifier(n_neighbors=1)
# model.fit(training_data, labels)
#
# # Test the model with a new image
# test_image = cv2.imread(r'C:\Users\JasonHinch\PycharmProjects\FacialRecognition2\nicolas-cage-test.png')
# test_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
# faces = face_cascade.detectMultiScale(test_gray, 1.3, 5)
# for (x, y, w, h) in faces:
#     face_img = test_gray[y:y+h, x:x+w]
#     face_img = cv2.resize(face_img, (100, 100))
#     prediction = model.predict([face_img.flatten()])
#     print(prediction)  # Output the prediction
#
# cv2.destroyAllWindows()