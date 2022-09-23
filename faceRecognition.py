import face_recognition
image = face_recognition.load_image_file(".\Fotos\04A355BC-EDB1-477A-84D2-7D9E200C9ABB.jpg")
face_locations = face_recognition.face_locations(image)