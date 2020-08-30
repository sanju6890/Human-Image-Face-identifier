import cv2 as cv

# loading the cascade
face_cascade = cv.CascadeClassifier('face_detector.xml')
img = cv.imread('image.jpeg')
image = cv.imread('image.jpeg')

# Detection of face
faces = face_cascade.detectMultiScale(img, 1.1, 4)

# Drawing the Rectangle around the face detected
for (x, y, w, h) in faces:
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv.ellipse(img, (x+w//2, y+h//2), (w//2, h//2), 0, 0, 360, (0, 255, 0), 4)

# Expoeting the result
cv.imwrite('Face_detected.png', img)
print('Successfully saved...')
cv.imshow('Image', image)
cv.waitKey(5000)
cv.imshow('Face detection', img)
cv.waitKey(5000)
cv.destroyAllWindows()

