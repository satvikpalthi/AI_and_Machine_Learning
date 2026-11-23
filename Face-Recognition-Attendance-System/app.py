
import cv2
import face_recognition
import numpy as np
import os
import pickle
import csv
from datetime import datetime

DATASET = "dataset"
MODEL = "models/face_encodings.pkl"
ATTENDANCE = "attendance"

os.makedirs("models", exist_ok=True)
os.makedirs(ATTENDANCE, exist_ok=True)


def train_faces():
    encodings = []
    names = []

    for person in os.listdir(DATASET):
        folder = os.path.join(DATASET, person)

        if not os.path.isdir(folder):
            continue

        for image_name in os.listdir(folder):
            path = os.path.join(folder, image_name)

            if not image_name.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):
                continue

            image = face_recognition.load_image_file(path)
            faces = face_recognition.face_encodings(image)

            if len(faces) == 1:
                encodings.append(faces[0])
                names.append(person)
                print("Trained:", person)

    if not encodings:
        print("No valid face images found!")
        return

    with open(MODEL, "wb") as file:
        pickle.dump(
            {"encodings": encodings, "names": names},
            file
        )

    print("Training completed!")


def mark_attendance(name):
    date = datetime.now().strftime("%Y-%m-%d")
    time = datetime.now().strftime("%H:%M:%S")

    file_path = f"{ATTENDANCE}/attendance_{date}.csv"

    if os.path.exists(file_path):
        with open(file_path, "r", newline="") as file:
            rows = list(csv.reader(file))

        if any(row and row[0] == name for row in rows[1:]):
            return

    with open(file_path, "a", newline="") as file:
        writer = csv.writer(file)

        if os.path.getsize(file_path) == 0:
            writer.writerow(["Name", "Date", "Time"])

        writer.writerow([name, date, time])

    print("Attendance marked:", name)


def recognize_faces():
    if not os.path.exists(MODEL):
        print("Train the model first!")
        return

    with open(MODEL, "rb") as file:
        data = pickle.load(file)

    known_encodings = data["encodings"]
    known_names = data["names"]

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Camera not available!")
        return

    while True:
        success, frame = camera.read()

        if not success:
            break

        small_frame = cv2.resize(
            frame, (0, 0), fx=0.25, fy=0.25
        )

        rgb = cv2.cvtColor(
            small_frame, cv2.COLOR_BGR2RGB
        )

        locations = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(
            rgb, locations
        )

        for encoding, location in zip(encodings, locations):
            name = "Unknown"

            if known_encodings:
                distances = face_recognition.face_distance(
                    known_encodings, encoding
                )

                best = np.argmin(distances)

                if distances[best] < 0.48:
                    name = known_names[best]
                    mark_attendance(name)

            top, right, bottom, left = location

            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

            cv2.rectangle(
                frame, (left, top), (right, bottom), color, 2
            )

            cv2.putText(
                frame, name,
                (left, bottom + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, color, 2
            )

        cv2.imshow("Face Recognition Attendance", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("FACE RECOGNITION ATTENDANCE SYSTEM")
    print("1. Train Model")
    print("2. Start Attendance")

    choice = input("Enter choice: ")

    if choice == "1":
        train_faces()
    elif choice == "2":
        recognize_faces()
    else:
        print("Invalid choice!")
