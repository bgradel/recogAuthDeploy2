import cv2
import glob
import os
from retinaface import RetinaFace
from wand.image import Image
import numpy as np
import math
   


def getFaceFromFacialRegion(facial_area, img):
    '''
    Helper function which returns the face image based off of it's given area from the original image
    
    Parameters:
        facial_area: List of integers
        img: opencv image
    Returns:
        opencv image containing the face
    '''
    y = facial_area[1]
    h = facial_area[3] - y
    x = facial_area[0]
    w = facial_area[2] - x
    img_region = [x, y, w, h]
    
    return img[facial_area[1]: facial_area[3], facial_area[0]: facial_area[2]]

            
def cropFaces(file_name, min_height=100, min_width=100, write_faces=False, output_folder=""):
    '''
    Removes all faces from the given file using RetinaFace
    
    Parameters:
        file_name: str of the name of the image that you want to remove faces from
        min_height: int representing the minimum face image height to save
        min_height: int representing the minimum face image width to save
        write_faces: boolean for whether or not to save the image to a file
        output_folder: location to save face images to if desired
    Returns:
        list of opencv images containing the faces from the original images
    '''
    img = cv2.imread(file_name)
    base = os.path.basename(file_name).split('.')[0]
    
    face_detector = RetinaFace.build_model()
    obj = RetinaFace.detect_faces(img_path=file_name, model = face_detector, threshold = 0.5)
    
    faces = []
    n = 0
    if type(obj) == dict:
        for key in obj:
            identity = obj[key]

            facial_area = identity["facial_area"]
            
            detected_face = getFaceFromFacialRegion(facial_area, img)
            
            h, w, d = detected_face.shape
            
            if h >=min_height and w >= min_width:
            
                faces.append(detected_face)
                
                if write_faces:
                    face_output = output_folder + '/' + base + '_' +  str(n) + '.jpg'

                    os.makedirs(os.path.dirname(face_output), exist_ok=True)
                    cv2.imwrite(face_output, detected_face)
                    n+=1
    return faces
    
def cropAllFaces(img_folder, min_height=100, min_width=100, write_faces= False, output_folder=''):
    '''
    Removes all faces from every image in the given folder using RetinaFace
    
    Parameters:
        img_folder: str of the name of the folder containing the images that you want to remove faces from
        min_height: int representing the minimum face image height to save
        min_height: int representing the minimum face image width to save
        write_faces: boolean for whether or not to save the image to a file
        output_folder: location to save face images to if desired
    Returns:
        list of opencv images containing the faces from the original images
    '''
    img_pattern = img_folder + '/*.jpg'
    img_list=glob.glob(img_pattern)
    
    faces = []

    for img in img_list:
        faces.extend(cropFaces(img, min_height, min_width, write_faces, output_folder))
        
    return faces
    
    
def distortImageThroughEdgeDetection(file_name, write_img=False, output_folder=""):
    '''
    Distorts an image from a file and returns the distorted image through edge detection
    
    Parameters:
        file_name: str of the name of the image file that you want to distort
        write_img: boolean for whether or not to save the image to a file
        output_folder: location to save face images to if desired
    Returns:
        opencv image distorted by edge dectection
    '''
    img=cv2.imread(file_name)
    edges = cv2.Canny(img,50,150)
    
    if write_img:
        base = os.path.basename(file_name).split('.')[0]
        output_file = output_folder + '/' + base + '_edges.jpg'
        cv2.imwrite(output_file, edges)
    
    return edges
    

    
def distortImageThroughBlurring(file_name, write_img=False, output_folder=""):
    '''
    Distorts an image from a file and returns the distorted image through blurring the image
    
    Parameters:
        file_name: str of the name of the image file that you want to distort
        write_img: boolean for whether or not to save the image to a file
        output_folder: location to save face images to if desired
    Returns:
        opencv image distorted by bluring
    '''
    face = Image(filename =file_name)
    img=cv2.imread(file_name)
    
 
    face.blur(sigma = 8)
    
    if write_img:
        base = os.path.basename(file_name).split('.')[0]
        output_file = output_folder + '/' + base + '_blur.jpg'
        face.save(filename =output_file)
    
    img = changeMagicImageToOpenCV(face)
            
    return img
    
def distortImageThroughPolar(file_name, write_img=False, output_folder=""):
    '''
    Distorts an image from a file and returns the distorted image through blurring the image
    
    Parameters:
        file_name: str of the name of the image file that you want to distort
        write_img: boolean for whether or not to save the image to a file
        output_folder: location to save face images to if desired
    Returns:
        opencv image distorted by bluring
    '''
    face = Image(filename =file_name)
    img=cv2.imread(file_name)
    
    h, w, d = img.shape
    face.distort('polar', (-1,0, 0, 0, -90, 90))
    
    if write_img:
        base = os.path.basename(file_name).split('.')[0]
        output_file = output_folder + '/' + base + '_polar.jpg'
        face.save(filename =output_file)
    
    img = changeMagicImageToOpenCV(face)       
    return img
    
def changeMagicImageToOpenCV(img):
    '''
    Helper function which converts to an open cv image
    
    Parameters:
        img: the image to convert
    Returns:
        opencv image
    '''
    img_buffer=np.asarray(bytearray(img.make_blob()), dtype=np.uint8)

    if img_buffer is not None:
        img = cv2.imdecode(img_buffer, cv2.IMREAD_UNCHANGED)
        
    return img
    
def distortAllWithDistortions(folder, distortions=[], write_img=False, output_folders=[]):
    '''
    Distorts all images in the given folder using each of the distortions in the given list
    
    Parameters:
        folder = the folder of images to distort
        distortion = str specifying which distortion to apply
        write_img = boolean for whether to write the resulting images to a file
        output_folder = location for the new distorted images
    Returns:
        dictionary containing distorted opencv images
    '''
    img_pattern = folder + '/*.jpg'
    img_list=glob.glob(img_pattern)
    print(folder)
    print(img_list)
    
    distorted_imgs = {}
    
    n=0
    for distortion in distortions:
        distortions_list = []
        for img in img_list:
            if distortion == "edges":
                distortions_list.extend(distortImageThroughEdgeDetection(img, write_img, output_folders[n]))
            elif distortion == "blur":
                distortions_list.extend(distortImageThroughBlurring(img, write_img, output_folders[n]))
            elif distortion == "polar":
                distortions_list.extend(distortImageThroughPolar(img, write_img, output_folders[n]))
            else:
                print("specified distortion was not recognized. Please use 'edges', 'blur', or 'polor'")
        distorted_imgs[distortion] = distortions_list
        n+=1
               
    return distorted_imgs
    
    
distortAllWithDistortions("./user1/chaff/not_distorted", distortions=["edges", "blur", "polar"], write_img=True, output_folders=["./user1/chaff/edges", "./user1/chaff/blur", "./user1/chaff/polar"])
    






    
    
 

     


    
