import face_recognition
import cv2
import numpy as np

lebron_img = face_recognition.load_image_file("lebron.png")
lebron_encoding = face_recognition.face_encodings(lebron_img)[0]

wade_img = face_recognition.load_image_file("wade.webp")
wade_encoding = face_recognition.face_encodings(wade_img)[0]

bosh_img = face_recognition.load_image_file("bosh.webp")
bosh_encoding = face_recognition.face_encodings(bosh_img)[0]

haslem_img = face_recognition.load_image_file("haslem.webp")
haslem_encoding = face_recognition.face_encodings(haslem_img)[0]


known_face_encodings = [
    lebron_encoding,
    wade_encoding,
    bosh_encoding,
    haslem_encoding
]
known_face_names = [
    "Lebron James",
    "Dwyane Wade",
    "Chris Bosh",
    "Udonis Haslem"
]

face_locations = []
face_encodings = []
face_names = []



frame = face_recognition.load_image_file("heat.jpg")
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

face_locations = face_recognition.face_locations(frame)
face_encodings = face_recognition.face_encodings(frame, face_locations)

for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # # If a match was found in known_face_encodings, just use the first one.
        # if True in matches:
        #     first_match_index = matches.index(True)
        #     name = known_face_names[first_match_index]

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)


for (top, right, bottom, left), name in zip(face_locations, face_names):
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

cv2.imshow("frame", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()







# face_locations = face_recognition.face_locations(image)

# image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# # Draw boxes around the faces
# for face_location in face_locations:
#     top, right, bottom, left = face_location
#     cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

# cv2.imshow("Faces", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# print(face_locations)