

from scipy.io import loadmat
import csv
import ast
import numpy as np
from array import array
from PIL import Image
import cv2
import os

#INPUT
# input_mat = loadmat("/project/ntimea/NewData/MATFILES/NewCut_FINAL_ALL2.mat")
# osszesitett_data = input_mat["SobelStruct"]
filename2 = "/project/ntimea/NewData/MATFILES/NewAugment/NewData/3D_SOLD_seq05.csv"
filename3 = "/project/ntimea/NewData/MATFILES/Data_img_SOLD_seq05.csv"
imagepath = "/project/Datasets/KITTI_360/2013_05_28_drive_0005_sync/image_00/data_rect/"


#OUTPUT
mozaikarraypath = '/project/ntimea/NewData/MATFILES/NewAugment/NewData/Mozaik4_SOLD_seq5.csv'
mozaikpath = "/project/ntimea/NewData/IMAGES/ImageProcessing/Mozaik4/Images_SOLD_seq5/"
mozaikLinesPath = "/project/ntimea/NewData/IMAGES/ImageProcessing/Mozaik4/Lines_seq5/"

if not os.path.exists(mozaikpath):
    os.makedirs(mozaikpath)
    os.makedirs(mozaikLinesPath)

def read_csv(file_path):
    # Read the CSV file
    with open(file_path, 'r') as csvfile:
        # Create a CSV reader object
        csvreader = csv.reader(csvfile)
        # Get the header
        header = next(csvreader)
        # Create an empty list to store the csvdata
        csvdata = []
        # Iterate over each row in the csv file
        for row in csvreader:
            # Create an empty dictionary
            row_dict = {}
            # Iterate over each cell in the row and the corresponding header cell
            for cell, header_cell in zip(row, header):
                # Add the cell value to the dictionary with the header cell as the key
                row_dict[header_cell] = cell
            # Add the dictionary to the csvdata list
            csvdata.append([row_dict])
        # Return the csvdata list
        return csvdata
    
def write_data_to_csv(data_array, filename):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['ID','2D', 'id_3D', '3D'  ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for data in data_array:
            writer.writerow(data)


def have_common_elements(arr1, arr2):
    set1 = set(arr1)
    set2 = set(arr2)
    return bool(set1.intersection(set2))

# Gives back the first element in the array which does not have any common 3d lines
def notpair(notin_list):
    for data in range(len(NewCutted3DID)):
        if isinstance(NewCutted3DID[data], array):
            NewCutted3DID[data] = NewCutted3DID[data][0]
        checkdata = NewCutted3DID[data]
        if not (have_common_elements( checkdata["id_3D"] , notin_list)):
            if(checkdata["ID"] not in blacklistid):
                blacklistid.append(NewCutted3DID[data]["ID"])
                return NewCutted3DID[data]
        
    return "error"

def firstelem(blacklistid):
    for data in range(len(NewCutted3DID)):
        if(NewCutted3DID[data]["ID"] not in blacklistid):
            blacklistid.append(NewCutted3DID[data]["ID"])
            return NewCutted3DID[data]
    raise Exception ("Nothing good")


def stack_images_with_flipped_top_and_bottom(image_paths):
    images = [Image.open(path) for path in image_paths]

    # # Flip the first and last images vertically
    images[0] = images[0].transpose(Image.FLIP_TOP_BOTTOM)
    images[2] = images[2].transpose(Image.FLIP_TOP_BOTTOM)

    # Calculate tot2l height for the new image
    total_height = sum(img.size[1] for img in images)

    # Calculate maximum width among all images
    max_width = max(img.size[0] for img in images)

    # Create a new image with white background
    new_image = Image.new("RGB", (max_width, total_height), color="white")

    # Paste each image onto the new image
    y_offset = 0
    for img in images:
        new_image.paste(img, (0, y_offset))
        y_offset += img.size[1]

    return new_image


def changeimagename2(original_filename):
    numerical_part = original_filename.split("_")[-1].split(".")[0]

    return numerical_part


def changeimagename(original_filename):
    # Remove non-numeric characters
    numeric_str = ''.join(filter(str.isdigit, original_filename))

    # Convert the numeric string to an integer
    result = int(numeric_str)

    # Format the result with leading zeros
    formatted_result = f"{result:010d}"

    return formatted_result

def flip_line(image_height, line):
    x1 = line[0]
    y1 = line[1]
    x2 = line[2]
    y2 = line[3]


    new_y1 = image_height - y1
    new_y2 = image_height - y2
    return x1, new_y1, x2, new_y2

def pushdownflip(i, image_lines, image_height):
    newlines = []
    for line in image_lines:
        #for line in image:
        x1 = line[0]
        y1 = line[1]
        x2 = line[2]
        y2 = line[3]

        if i == 0 or i == 2:
            new_y1 = 376 - y1
            new_y2 = 376 - y2
        else:
            new_y1 = y1
            new_y2 = y2

        new_y1 = new_y1 + i*image_height
        new_y2 = new_y2 + i*image_height




        newlines.append([x1, new_y1, x2, new_y2])
    
    return newlines


def displaylines(inputpath, outputpath, allines):
    
    image_path = inputpath 
    img = cv2.imread(image_path)

    for lines in allines:
        for line in lines:
            x1 = int(line[0])
            y1 = int(line[1])
            x2 = int(line[2])
            y2 = int(line[3])
            img = cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 4)

    
    cv2.imwrite(outputpath, img)

def write_data_to_csv(data_array, filename, fieldnames):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for data in data_array:
            writer.writerow(data)










newCutted3D = read_csv(filename2)
NewCutted3DID = read_csv(filename3)

for i in range(len(NewCutted3DID)):
    NewCutted3DID[i] = NewCutted3DID[i][0] 
    # NewCutted3DID[i]["ID"] = ast.literal_eval(NewCutted3DID[i]["ID"])
    NewCutted3DID[i]["2D"] = ast.literal_eval(NewCutted3DID[i]["2D"])
    NewCutted3DID[i]["id_3D"] = ast.literal_eval(NewCutted3DID[i]["id_3D"])
    NewCutted3DID[i]["3D"] = ast.literal_eval(NewCutted3DID[i]["3D"])
    
    

for i in range(len(newCutted3D)):
    newCutted3D[i] = newCutted3D[i][0]
    newCutted3D[i]["id_3D"] = newCutted3D[i]["id_3D"]
    newCutted3D[i]["3D"] = ast.literal_eval(newCutted3D[i]["3D"])
    newCutted3D[i]["2D"] = ast.literal_eval(newCutted3D[i]["2D"])


allnewimages = []
blacklistid = []


while len(blacklistid) <= len(NewCutted3DID):
    print(len(blacklistid), "------", len(NewCutted3DID))

    # if len(blacklistid) == 100:
    #     break

    notin_list = []
    images4 = []
    image = firstelem(blacklistid)
    #image = NewCutted3DID[2]
    images4.append(image)

    if not isinstance(image, dict):
        break

    for fourimage in range(4):
        if not isinstance(image, dict):
            break
        notin_list.extend(image["id_3D"])
        image = notpair(notin_list)
        images4.append(image)

    if not isinstance(image, dict):
        break


    if len(images4)>0:
        allnewimages.append(images4)



mozaikimageid = 0
allmozaikarray = []
for mozaikimage in allnewimages:
    mozaikimageid = mozaikimageid + 1
    mozaikimagenames = []
    mozaik2d = []
    mozaik2d_NEW = []
    mozaik3d = []
    mozaikID3d = []

    for i in range(4):
        mozaikimagenames.append(changeimagename(mozaikimage[i]["ID"]))
        mozaik2d.append((mozaikimage[i]["2D"]))
        mozaik3d.append((mozaikimage[i]["3D"]))
        mozaikID3d.append((mozaikimage[i]["id_3D"]))
        
        line = (mozaikimage[i]["2D"])
       
        newlines = pushdownflip(i, line, 376)
        mozaik2d_NEW.append(newlines)


    image_paths = [imagepath + mozaikimagenames[0][1:] +".png" , imagepath + mozaikimagenames[1][1:] +".png", imagepath + mozaikimagenames[2][1:] +".png", imagepath + mozaikimagenames[3][1:] +".png"] # Replace with your image paths
    new_image = stack_images_with_flipped_top_and_bottom(image_paths)
    imagepath2 = (mozaikpath+ f"{mozaikimageid}.jpg")

    outputpath = (mozaikLinesPath+f"{mozaikimageid}.jpg")
    new_image.save(imagepath2)
    displaylines(imagepath2, outputpath, mozaik2d_NEW)
    
    #CREATE DICTIONARY
    mozaikdictionary = {}
    mozaikdictionary["ID"] = mozaikimageid
    mozaikdictionary["2D"] = mozaik2d_NEW
    mozaikdictionary["3D"] = mozaik3d
    mozaikdictionary["id_3D"] = mozaikID3d
    mozaikdictionary["ID_orig"] = mozaikimagenames
    mozaikdictionary["2D_orig"] = mozaik2d


    allmozaikarray.append(mozaikdictionary)


    

fieldnames = ['ID','2D', '3D', 'id_3D', 'ID_orig', '2D_orig' ]

write_data_to_csv(allmozaikarray, mozaikarraypath, fieldnames)

    

   
   


    


















    

    




