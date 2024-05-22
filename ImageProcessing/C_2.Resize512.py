from scipy.io import savemat
import scipy.io
import numpy as np
from PIL import Image, ImageDraw
import math
import matplotlib.pyplot as plt
import scipy.io as sio
import csv
import ast
import cv2
import os

#INPUT
#data = scipy.io.loadmat('/Users/timeanemet/Desktop/CNN/matfiles/subset_data.mat')
FilePath = '/project/ntimea/NewData/MATFILES/NewAugment/NewData/Cutted_Data_img_SOLD_seq05.csv'
imagePath ="/project/Datasets/KITTI_360/2013_05_28_drive_0005_sync/image_00/data_rect/"
#OUTPUT
OutPath = '/project/ntimea/NewData/IMAGES/ImageProcessing/Cutted_Data_img_SOLD_seq05/'



def read_csv(file_path):
    # Read the CSV file
    with open(file_path, 'r') as csvfile:
        # Create a CSV reader object
        csvreader = csv.reader(csvfile)
        # Get the header
        header = next(csvreader)
        # Create an empty list to store the array_3D
        array_3D = []
        # Iterate over each row in the csv file
        countrow = 0
        for row in csvreader:
            countrow = countrow+1
            # Create an empty dictionary
            row_dict = {}
            # Iterate over each cell in the row and the corresponding header cell
            for cell, header_cell in zip(row, header):
                # Add the cell value to the dictionary with the header cell as the key
                row_dict[header_cell] = cell
            # Add the dictionary to the array_3D list
            array_3D.append([row_dict])
        # Return the array_3D list
        return array_3D


def resizedlinesGT(OutPath, allines, old_size, new_size):
    newlines = []

    img = np.zeros(new_size, dtype=np.uint8)

    # Calculate the scaling factor
    scale_factor = new_size[0] / old_size[0]
    scale_factor2 = new_size[1] / old_size[1]

    for lines in allines:
        # Scale the coordinates
        x1 = int(lines[0] * scale_factor2)
        y1 = int(lines[1] * scale_factor)
        x2 = int(lines[2] * scale_factor2)
        y2 = int(lines[3] * scale_factor)

        # Draw the line on the resized image
        img = cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
        newlines.append([x1, y1, x2, y2])
       

    imagePath = OutPath
    cv2.imwrite(imagePath, img)

    return newlines


def displaylinesonimage(InPath, OutPath, allines, cuttedhere, pt=4):
    img = cv2.imread(InPath)
    width, height = 376, 1408

    x1_verti = cuttedhere[0]
    x2_verti = cuttedhere[1]


    for lines in allines:
        
        x1 = int(lines[0])
        y1 = int(lines[1])
        x2 = int(lines[2])
        y2 = int(lines[3])
        img = cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), pt)


    cv2.line(img, (x1_verti, 0), (x1_verti, height), (255, 0, 0), pt)
    cv2.line(img, (x2_verti, 0), (x2_verti, height), (255,0, 0), pt)    

    cv2.imwrite(OutPath, img)


def resize_and_draw_lines(InPath, OutPath, allines, pt=4):
    # Read the image from the input path
    if InPath != None:
        img = cv2.imread(InPath)
        color = (0, 0, 255)
    else:
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        color = (255, 255, 255)

    

    # Ensure the image is read correctly
    if img is None:
        print(f"Error: Unable to read the image from {InPath}")
        return

    # Resize the image from 376x376 to 512x512
    original_size = (376, 376)
    new_size = (512, 512)
    img_resized = cv2.resize(img, new_size)

    # Calculate the scaling factor
    scale_x = new_size[0] / original_size[0]
    scale_y = new_size[1] / original_size[1]

    # Adjust and draw lines on the resized image
    for lines in allines:
        x1 = int(lines[0] * scale_x)
        y1 = int(lines[1] * scale_y)
        x2 = int(lines[2] * scale_x)
        y2 = int(lines[3] * scale_y)
        img_resized = cv2.line(img_resized, (x1, y1), (x2, y2), color, pt)

    # Save the processed image to the output path
    cv2.imwrite(OutPath, img_resized)


def resizeimg(InPath, OutPath):

    if InPath != None:
        img = cv2.imread(InPath)
    else:
        img = np.zeros((512, 512, 3), dtype=np.uint8)


    # Resize the image from 376x376 to 512x512
    original_size = (376, 376)
    new_size = (512, 512)
    img_resized = cv2.resize(img, new_size)

    cv2.imwrite(OutPath, img_resized)

def drawGT(OutPath, allines, pt=4):
    # Create a black image
    img = np.zeros((512, 512, 3), dtype=np.uint8)

    # Adjust and draw lines on the black image
    for lines in allines:
        x1 = int(lines[0])
        y1 = int(lines[1])
        x2 = int(lines[2])
        y2 = int(lines[3])
        img = cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), pt)

    # Save the processed image to the output path
    cv2.imwrite(OutPath, img)

csvdata = read_csv(FilePath)

def cutimage(InPath, OutPath, allines, cuttedhere, pt=4):
    img = cv2.imread(InPath)
    width, height = 376, 1408

    x1_verti = cuttedhere[0]
    x2_verti = cuttedhere[1]

    img = img[:, x1_verti:x2_verti]  # from x1 to x2

    # for lines in allines:
        
    #     x1 = int(lines[0])
    #     y1 = int(lines[1])
    #     x2 = int(lines[2])
    #     y2 = int(lines[3])
    #     img = cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), pt)

    cv2.imwrite(OutPath, img)



for i in range(len(csvdata)):
    csvdata[i] = csvdata[i][0]
    csvdata[i]["ID"] = csvdata[i]["ID"]
    csvdata[i]["id_3D"] = ast.literal_eval(csvdata[i]["id_3D"])
    csvdata[i]["2D_orig"] = ast.literal_eval(csvdata[i]["2D_orig"])
    csvdata[i]["2D"] = ast.literal_eval(csvdata[i]["2D"])
    csvdata[i]["3D"] = ast.literal_eval(csvdata[i]["3D"])
    csvdata[i]["cutedhere"] = ast.literal_eval(csvdata[i]["cutedhere"])


# ['ID', '2D', '2D_orig', '3D', 'cutedhere']

#resizedlinesGT(OutPath3,lines, old_size=(376, 376), new_size=(512, 512))

allaray = []
dicty = {}

if not os.path.exists(OutPath):
    os.makedirs(OutPath)

GTPATH = OutPath + "GT_512/"
IMGPATH = OutPath + "ImagesLINES_512/"
Image_512 =  OutPath + "Images_512/"


if not os.path.exists(GTPATH):
    os.makedirs(GTPATH)
    os.makedirs(IMGPATH)
    os.makedirs(Image_512)


for i in range(len(csvdata)):

    print(i, "----", len(csvdata))
    id = csvdata[i]["ID"]
    number = id.split('_')[1]
    id  = number.replace('[', '').replace(']', '')[:-1]
    data2d_orig = csvdata[i]["2D_orig"]
    data2d = csvdata[i]["2D"]
    cuttedhere = csvdata[i]["cutedhere"]

   #
    impath = id
    image_id = imagePath + impath + ".png"

    OutPath_img = OutPath + impath + ".png"
    OutPath_cut = OutPath + impath + "_cut.png"

    #print(cuttedhere[1]-cuttedhere[0])
    number  = csvdata[i]["ID"][-2:] 




    displaylinesonimage(image_id, OutPath_img, data2d_orig, cuttedhere, pt=4)
    OutPath_cut = OutPath + impath  + number+".png"


    cutimage(image_id, OutPath_cut, data2d, cuttedhere, pt=4)

    OutPath_512_LINES =  IMGPATH + impath  + number + ".png"

    


    resize_and_draw_lines(OutPath_cut, OutPath_512_LINES, data2d, pt=4)

    OutPath_512GT = GTPATH + impath  + number + ".png"

   

    resize_and_draw_lines(None, OutPath_512GT, data2d, pt=4)

    OutPath_512 = Image_512 + impath  + number + ".png"

    resizeimg(OutPath_cut, OutPath_512)



    dicty = {}
    dicty["ID"] = id
    dicty["2D"] = data2d

    allaray.append(dicty)











def write_data_to_csv(data_array, filename):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['ID', '2D']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for data in data_array:
            writer.writerow(data)


#write_data_to_csv(allaray, FilePath)