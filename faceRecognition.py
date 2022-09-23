import face_recognition
import os
import shutil


def clusterPhotosByFace(myPhotoPath,targetDir,resultsDir,nomatchDir):
    ## Get all the target photos file names in a list
    photosList = os.listdir(targetDir)


    ## Get face encodings of my photo
    myImage = face_recognition.load_image_file(myPhotoPath)
    myFaceEncoding = face_recognition.face_encodings(myImage)[0]

    faceLocations = []
    faceEncodings = []
    matches = []
    ## Iterate through the photosList
    for photo in photosList:

        ## Get photo face locations and face encodings
        targetPhoto = face_recognition.load_image_file(targetDir + '/' + photo)
        faceLocations = face_recognition.face_locations(targetPhoto)
        faceEncodings = face_recognition.face_encodings(targetPhoto, faceLocations)

        ## Iterate through face encodings to compare to myFaceEncoding
        for targetEncoding in faceEncodings:
            ## By default compare_faces needs a list as a parameter, for testing purposes we are using the same image twice as a list
            matches = face_recognition.compare_faces([myFaceEncoding,myFaceEncoding], targetEncoding)

            ## If True then move the photo from root to results directory and break the loop (since we are only looking for a single face, one encoding match is enough)
            if True in matches:
                shutil.move(targetDir+'/'+photo, resultsDir+'/'+photo)
                break
        # If for loop was finished (not break) and last matches doesn't have True, move the photo to another directory
        if not True in matches:
            shutil.move(targetDir+'/'+photo, nomatchDir+'/'+photo)


    print("Finished moving photos")


##Photos paths
myPhotoPath = "./mypicture/me.jpg"
targetDir = "./photos"
resultsDir = "./results"
nomatchDir = "./nomatch"

##Make sure the resultsDir folder exists before running the script
clusterPhotosByFace(myPhotoPath,targetDir,resultsDir,nomatchDir)
