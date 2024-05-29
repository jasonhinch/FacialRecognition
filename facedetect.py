import cv2
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier

# Load images
image_paths = [r'C:\Users\JasonHinch\PycharmProjects\FacialRecognition2\Nicolas_Cage.webp', r'C:\Users\JasonHinch\PycharmProjects\FacialRecognition2\the-15-best-nicolas-cage-movies_bvu1.1280.webp']  # Add paths to Nicholas Cage images
training_data = []
labels = []

# Haar Cascade Classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Process each image
for image_path in image_paths:
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (100, 100))
        training_data.append(face_img.flatten())
        labels.append("Nicholas Cage")

# Train KNN classifier
model = KNeighborsClassifier(n_neighbors=1)
model.fit(training_data, labels)

# Test the model with a new image
test_image = cv2.imread(r'C:\Users\JasonHinch\PycharmProjects\FacialRecognition2\nicolas-cage-test.png')
test_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(test_gray, 1.3, 5)
for (x, y, w, h) in faces:
    face_img = test_gray[y:y+h, x:x+w]
    face_img = cv2.resize(face_img, (100, 100))
    prediction = model.predict([face_img.flatten()])
    print(prediction)  # Output the prediction

cv2.destroyAllWindows()




# import cv2
# import numpy as np
# import npwriter
#
# name = input("Enter your name: ")
# cap = cv2.VideoCapture(0)
# classifier = cv2.CascadeClassifier(r'C:\Users\JasonHinch\PycharmProjects\FacialRecognition2\haarcascade_frontalface_default.xml')
# f_list = []
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         continue  # This continue must be inside the loop
#
#     # converting the image into gray scale as it is easy for detection
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     # detect multiscale, detects the face and its coordinates
#     faces = classifier.detectMultiScale(gray, 1.5, 5)
#     # this is used to detect the face which is closest to the web-cam on the first position
#     faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
#     # only the first detected face is used
#     faces = faces[:1]
#
#     if len(faces) == 1:
#         face = faces[0]
#         x, y, w, h = face
#         im_face = frame[y:y + h, x:x + w]
#         cv2.imshow("face", im_face)  # This line should be inside the loop
#
#     cv2.imshow("full", frame)
#     key = cv2.waitKey(1)
#
#     if key & 0xFF == ord('q'):
#         break  # This break must be inside the loop
#     elif key & 0xFF == ord('c'):
#         if len(faces) == 1:
#             gray_face = cv2.cvtColor(im_face, cv2.COLOR_BGR2GRAY)
#             gray_face = cv2.resize(gray_face, (100, 100))
#             f_list.append(gray_face.reshape(-1))
#             if len(f_list) == 10:
#                 break  # This break must be inside the loop
#         else:
#             print("Face not found")
#
# # Store the data after exiting the loop
# npwriter.write(name, np.array(f_list))
# cap.release()
# cv2.destroyAllWindows()