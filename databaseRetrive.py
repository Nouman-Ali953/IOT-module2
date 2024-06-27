import base64
import cv2
import face_recognition
import pickle
import pymongo
import numpy as np

# Connect to MongoDB
try:
    # client = pymongo.MongoClient("mongodb+srv://nauman:fypcluster@cluster0.ascwljq.mongodb.net/?retryWrites=true&w=majority")
    client = pymongo.MongoClient("mongodb://localhost:27017/fyp")
    db = client["fyp"]
    collection = db["people"]

    # Check if the connection is successful
    if client is not None and db is not None:
        print("MongoDB Connection Successful")
    else:
        print("MongoDB Connection Failed")

except Exception as e:
    print(f"Error connecting to MongoDB: {e}")

# Function to retrieve images from MongoDB
def get_images_from_mongodb():
    img_list = []
    student_info_list = []

    for document in collection.find():
        # Decode Base64 string to bytes
        base64_image_data = document["image"].split(',')[1]
        image_data = base64.b64decode(base64_image_data)
        img = cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_COLOR)

        student_info = {
            "_id": document["_id"],
            "name": document["name"],
            "designation": document.get("designation", "N/A"),
            "years_experience": document.get("years_experience", "N/A"),
            "filename": document.get("filename", "N/A")
        }

        img_list.append(img)
        student_info_list.append(student_info)

    return img_list, student_info_list

# Importing student images from MongoDB
images_list, student_info_list = get_images_from_mongodb()

def findEncodings(images_list):
    encode_list = []
    for img in images_list:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img)
        if encodings:  # Check if the list is not empty
            encode_list.append(encodings[0])
        else:
            print("No face found in an image")
    return encode_list

print("Encoding Started ...")
encode_list_known = findEncodings(images_list)

# Combine encodings with student information
encoded_data = []
for encode, info in zip(encode_list_known, student_info_list):
    encoded_info = {
        "encoding": encode,
        "_id": info["_id"],
        "name": info["name"],
        "designation": info["designation"],
        "years_experience": info["years_experience"],
        "filename": info["filename"]
    }
    encoded_data.append(encoded_info)

print("Encoding Complete")

# Save the encodings and student info to a file
file_path = "EncodeFile.p"
with open(file_path, 'wb') as file:
    pickle.dump(encoded_data, file)

print("File Saved")
