import face_recognition
import os
import shutil
import cv2


def clusterPhotosByFace(myPhotoPath,targetDir,resultsDir,nomatchDir):
    ## Get all the target photos file names in a list
    photosList = os.listdir(targetDir)


    ## Get face encodings of my photo
    myImage = face_recognition.load_image_file(myPhotoPath)
    myFaceEncoding = face_recognition.face_encodings(myImage)[0]

    faceLocations = []
    faceEncodings = []
    ## Iterate through the photosList
    for photo in photosList:

        ## Use opencv to read the image and resize it to 1/4 size for faster processing
        targetPhoto = cv2.imread(targetDir + '/' + photo)
        targetPhotoResized = cv2.resize(targetPhoto, (0, 0), fx=0.25, fy=0.25)

        ## Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        targetPhotoResized_RGB = targetPhotoResized[:, :, ::-1]
        
        ## Get photo face locations and face encodings using CNN model
        faceLocations = face_recognition.face_locations(targetPhotoResized_RGB,1,'cnn')
        faceEncodings = face_recognition.face_encodings(targetPhotoResized_RGB, faceLocations)
        
        ## Iterate through face encodings to compare to myFaceEncoding
        for targetEncoding in faceEncodings:

            ## By default compare_faces needs a list as a parameter, for testing purposes we are using the same image twice as a list
            matches = face_recognition.compare_faces([myFaceEncoding,myFaceEncoding], targetEncoding)

            ## If True then move the photo from root to results directory and break the loop (since we are only looking for a single face, one encoding match is enough)
            if True in matches:
                shutil.move(targetDir+'/'+photo, resultsDir+'/'+photo)
                break
            else:
                shutil.move(targetDir+'/'+photo, nomatchDir+'/'+photo)
        
    print("Finished moving photos")


##Photos paths
myPhotoPath = "./mypicture/me.jpg"
targetDir = "./photos"
resultsDir = "./results"
nomatchDir = "./nomatch"

##Make sure the resultsDir folder exists before running the script
clusterPhotosByFace(myPhotoPath,targetDir,resultsDir,nomatchDir)
