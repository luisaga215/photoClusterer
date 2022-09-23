import face_recognition
import os
import shutil


def clusterPhotosByFace(myPhotoPath,targetPhotosDir,resultsDir):
    ##Get all the target photos file names in a list
    photosList = os.listdir(targetPhotosDir)


    ##Get face encodings of my photo
    myImage = face_recognition.load_image_file(myPhotoPath)
    myFaceEncoding = face_recognition.face_encodings(myImage)[0]

    ##Iterate through the photosList
    for photo in photosList:

        ##Get photo face locations and face encodings
        targetPhoto = face_recognition.load_image_file(targetPhotosDir + '/' + photo)
        faceLocations = face_recognition.face_locations(targetPhoto)
        faceEncodings = face_recognition.face_encodings(targetPhoto, faceLocations)

        ##Iterate through face encodings to compare to myFaceEncoding
        for targetEncoding in faceEncodings:
            ##By default compare_faces needs a list as a parameter, for testing purposes we are using the same image twice as a list
            matches = face_recognition.compare_faces([myFaceEncoding,myFaceEncoding], targetEncoding)

            ##If True then move the photo from root to results directory and break the loop (since we are only looking for a single face, one encoding match is enough)
            if True in matches:
                shutil.move(targetPhotosDir+'/'+photo, resultsDir+'/'+photo)
                break

    print("Finished moving photos")


##Photos paths
myPhotoPath = "./mypicture/me.jpg"
targetPhotosDir = "./photos"
resultsDir = "./results"

##Make sure the resultsDir folder exists before running the script
clusterPhotosByFace(myPhotoPath,targetPhotosDir,resultsDir)
