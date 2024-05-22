


import scipy.io
import csv
import ast
import numpy as np
import cv2
from PIL import Image
import os

# INPUT
folder_path = "/project/Datasets/KITTI_360/2013_05_28_drive_0005_sync/image_00/data_rect/"
FileNameImage = "/project/ntimea/NewData/MATFILES/Data_img_SOLD_seq05.csv"
# TESTIMAGE = "/project/ntimea/NewData/CODES/StackImages_forOldData/Stack2Image/TESTIMAGE/IMAGES/"
# TESTIMAGE_Lines = "/project/ntimea/NewData/CODES/StackImages_forOldData/Stack2Image/TESTIMAGE/LINES/"
# TESTIMAGE_GT= "/project/ntimea/NewData/CODES/StackImages_forOldData/Stack2Image/TESTIMAGE/GT/"
# TESTIMAGE_GT_512 = "/project/ntimea/NewData/CODES/StackImages_forOldData/Stack2Image/TESTIMAGE/GT_512/"
# TESTIMAGE_IMAGES_512 = "/project/ntimea/NewData/CODES/StackImages_forOldData/Stack2Image/TESTIMAGE/IMAGES_512/"
# TESTIMAGE_LINES_512 = "/project/ntimea/NewData/CODES/StackImages_forOldData/Stack2Image/TESTIMAGE/LINES_512/"

TESTIMAGEBIG = "/project/ntimea/NewData/IMAGES/ImageProcessing/Mozaik2/BigImage_SOLD_05/"
TESTIMAGE = "/project/ntimea/NewData/IMAGES/ImageProcessing/Mozaik2/Images_SOLD_05/"
TESTIMAGE_Lines = "/project/ntimea/NewData/IMAGES/ImageProcessing/Mozaik2/Lines_SOLD_05/"
TESTIMAGE_GT= "/project/ntimea/NewData/IMAGES/ImageProcessing/Mozaik2/GT_SOLD_05/"
TESTIMAGE_GT_512 = "/project/ntimea/NewData/IMAGES/ImageProcessing/Mozaik2/GT_512_SOLD_05/"
TESTIMAGE_IMAGES_512 = "/project/ntimea/NewData/IMAGES/ImageProcessing/Mozaik2/Images_512_SOLD_05/"
TESTIMAGE_LINES_512 = "/project/ntimea/NewData/IMAGES/ImageProcessing/Mozaik2/Lines_512_SOLD_05/"
csvpath = "/project/ntimea/NewData/MATFILES/NewAugment/NewData/Mozaik2_SOLD_05.csv"




if not os.path.exists(TESTIMAGE):
    os.makedirs(TESTIMAGE)
    os.makedirs(TESTIMAGE_GT)
    os.makedirs(TESTIMAGE_Lines)
    os.makedirs(TESTIMAGE_GT_512)
    os.makedirs(TESTIMAGE_IMAGES_512)
    os.makedirs(TESTIMAGE_LINES_512)
    os.makedirs(TESTIMAGEBIG)

# OUTPUT


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
    
#ID,id_2D,id_3D,2D,3D,PosetoWorld,PosetoCamera,CalibMatrix
def arrangearray(csv_data):
    for i in range(len(csv_data)):
        csv_data[i] = csv_data[i][0]
        csv_data[i]["ID"] = csv_data[i]["ID"]
        csv_data[i]["id_3D"] = ast.literal_eval(csv_data[i]["id_3D"])
        csv_data[i]["2D"] = ast.literal_eval(csv_data[i]["2D"])
        csv_data[i]["3D"] = ast.literal_eval(csv_data[i]["3D"])
        




def write_data_to_csv(data_array, filename, fieldnames):
   with open(filename, 'w', newline='') as csvfile:
       fieldnames = fieldnames
       writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
       writer.writeheader()
       for data in data_array:
           writer.writerow(data)

def firstimage(BlackList, noLineOnimage):
    for i in range(len(images_csv_data)):
        if(images_csv_data[i]["ID"] not in BlackList):
            BlackList.append(images_csv_data[i]["ID"])
            if (len(images_csv_data[i]["2D"]) == 0):
                noLineOnimage = noLineOnimage+1
                continue
            else:
                return images_csv_data[i], BlackList, noLineOnimage
    BlackList_old = BlackList
    #Just to make the previous while stop
    while len(BlackList)-2 < len(images_csv_data):
        BlackList.append([])
    print("No more image", len(BlackList_old), len(images_csv_data))
    data2 = []
    return data2, BlackList, noLineOnimage

def same3d(search1, search2):
    for i in range(len(search1["id_3D"])):
        for j in range(len(search2["id_3D"])):
            if search1["id_3D"][i] == search2["id_3D"][j]:
                return True
    return False

def second(BlackList, noLineOnimage, search1):
    for i in range(len(images_csv_data)):
        if(images_csv_data[i]["ID"] not in BlackList):
            search2 = images_csv_data[i]
            if(same3d(search1, search2)) == False:
                BlackList.append(search2["ID"])
                if (len(search2["2D"]) == 0):
                    noLineOnimage = noLineOnimage+1
                    continue
                else:
                    return search2, BlackList, noLineOnimage
    while len(BlackList)-2 < len(images_csv_data):
        BlackList.append([])
    print("No more image", len(BlackList), len(images_csv_data))
    search2 = []
    return search2, BlackList, noLineOnimage

def displaylinesonimage_search(InPath, OutPath, Data):

    parts = Data["ID"][0].split("_", 1) 
    name = parts[1]

    img = cv2.imread(InPath + name + ".png")
    allines = Data["2D"]
    name = Data["ID"]
    for lines in allines:
        
        x1 = int(lines[0])
        y1 = int(lines[1])
        x2 = int(lines[2])
        y2 = int(lines[3])
        img = cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 4)

    cv2.imwrite(OutPath + name[0] + ".png",  img)




def displaylinesonimage(InPath, OutPath, allines, pt=4):
    img = cv2.imread(InPath)
    for lines in allines:
        
        x1 = int(lines[0])
        y1 = int(lines[1])
        x2 = int(lines[2])
        y2 = int(lines[3])
        img = cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), pt)

    cv2.imwrite(OutPath, img)


def displaylinesonimage_GT(OutPath, allines, size=(752, 704, 3)):
    # Create a black image
    img = np.zeros(size, dtype=np.uint8)

    for lines in allines:
        x1 = int(lines[0])
        y1 = int(lines[1])
        x2 = int(lines[2])
        y2 = int(lines[3])
        img = cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 4)

    cv2.imwrite(OutPath, img)



def stack_and_split_images(InPath, OutPath, search1, search2, Number):

    search1data = ast.literal_eval(search1["ID"])
    search1data = search1data[0]
    search1_split = search1data.split("_", 1) 
    search1_id = search1_split[1]
    search1_path = InPath + search1_id + ".png"

    search2data = ast.literal_eval(search2["ID"])
    search2data = search2data[0]
    search2_split = search2data.split("_", 1)  
    search2_id = search2_split[1]
    search2_path = InPath + search2_id + ".png"

    image1 = Image.open(search1_path)
    image2 = Image.open(search2_path)

    # Ensure the images have the same width
    width1, height1 = image1.size
    width2, height2 = image2.size
    new_width = max(width1, width2)

    # Create a new image with height equal to the sum of the two images' heights
    new_image = Image.new('RGB', (new_width, height1 + height2))

    # Paste the images into the new image
    new_image.paste(image1, (0, 0))
    new_image.paste(image2, (0, height1))

    # Save the new image
    new_image.save(TESTIMAGEBIG + str(Number) +"_"+ search1["ID"] + "_" +search2["ID"] + ".png")

    # Now, let's split the new image in half along the x-axis
    width, height = new_image.size
    half_width = width // 2

    # Create two new images from the halves
    image_left = new_image.crop((0, 0, half_width, height))
    image_right = new_image.crop((half_width, 0, width, height))

    # Save the new images
    image_left.save(OutPath + str(Number) + "_0.png")
    image_right.save(OutPath + str(Number) + "_1.png")

def leftright(search_line,left_imagelines, right_imagelines , left_imagelines3d, right_imagelines3d, search_line3d, left_3DID, right_3DID):
    Middleline = 704

    for i in range(len(search_line)):
        line = search_line[i]
        line3d = search_line3d[i]
        line3d_ID = search_line3d[i]
        x1 = line[0]
        y1 = line[1]
        x2 = line[2]
        y2 = line[3]
        if x1 > x2:
            x1_seg = x1
            x1 = x2
            x2 = x1_seg 

            y1_seg = y1
            y1 = y2
            y2 = y1_seg

        if x1 < Middleline and x2 < Middleline:
            left_imagelines.append(line)
            left_imagelines3d.append(line3d)
        elif x1 > Middleline and x2 > Middleline:
            line[0] = line[0]-Middleline
            line[2] = line[2]-Middleline
            right_imagelines.append(line)
            right_imagelines3d.append(line3d)

        else:
            #SLOPE
            if(x2-x1 != 0):
                slope = (y2-y1)/(x2-x1)
            else:
                slope = 0
            #CALC c (use x1,y1)
            c = (y1-(slope*x1))
            #CALC y (use middle)
            new_y = Middleline*(slope)+c

            lineleft = [x1,y1,Middleline,new_y]
            lineright = [0, new_y, x2-Middleline, y2]

            left_imagelines.append(lineleft)
            right_imagelines.append(lineright)

            left_imagelines3d.append(line3d)
            right_imagelines3d.append(line3d)

    return left_imagelines, right_imagelines, left_imagelines3d, right_imagelines3d,left_3DID, right_3DID

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

def resizeimage(InPath, OutPath, target_size=(512,512)):
    img = Image.open(InPath)
    img_resized = img.resize(target_size, Image.ANTIALIAS)
    img_resized.save(OutPath)

def movelines(search1,search2, OutPath, InPath, allimage, created_R, created_L,noimage ):

    thisimage = {}

    Middleline = 1408/2
    search1_line3d = search1["3D"]
    search1_line = search1["2D"]
    # PushDown
    new_search2 = search2
    for i in range(len(search2["2D"])):
        new_search2["2D"][i][1] = search2["2D"][i][1]+376
        new_search2["2D"][i][3] = search2["2D"][i][3]+376
    search2_line = new_search2["2D"]
    search2_line3d = search2["3D"]

    left_imagelines = []
    right_imagelines = []
    left_imagelines3d = []
    right_imagelines3d = []
    left_3d_ID, right_3d_ID = [], []

    #search1_line = [[10,500,1000,100]]

    InPath = TESTIMAGEBIG + str(Number) +"_"+ search1["ID"] + "_" +search2["ID"] + ".png"
    OutPath = TESTIMAGEBIG + str(Number) +"_"+ search1["ID"] + "_" +search2["ID"] + "_LINES.png"
    displaylinesonimage(InPath, OutPath, search1_line)
    displaylinesonimage(OutPath, OutPath, search2_line)


    left_imagelines, right_imagelines , left_imagelines3d, right_imagelines3d, left_3d_ID, right_3d_ID= leftright(search1_line, left_imagelines, right_imagelines,  left_imagelines3d, right_imagelines3d, search1_line3d, left_3d_ID, right_3d_ID)
    left_imagelines, right_imagelines,  left_imagelines3d, right_imagelines3d , left_3d_ID, right_3d_ID= leftright(search2_line, left_imagelines, right_imagelines,  left_imagelines3d, right_imagelines3d, search2_line3d, left_3d_ID, right_3d_ID)

    if len(left_imagelines) == 0:
        created_L = False
        print("NOLINES_L", Number, [search1["ID"], search2["ID"]])
        noimage = noimage + 1
    else:

        InPath = TESTIMAGE + f"{Number}_0.png"
        OutPath1 = TESTIMAGE_Lines + f"{Number}_0.png"
        OutPath2 = TESTIMAGE_GT + f"{Number}_0.png"
        OutPath3 = TESTIMAGE_GT_512 + f"{Number}_0.png"
        OutPath4 = TESTIMAGE_IMAGES_512 + f"{Number}_0.png"
        OutPath5 = TESTIMAGE_LINES_512 + f"{Number}_0.png"
        displaylinesonimage(InPath, OutPath1, left_imagelines, pt=4)
        displaylinesonimage_GT(OutPath2, left_imagelines)
        # 512
        resized_left_imagelines = resizedlinesGT(OutPath3,left_imagelines, old_size=(752, 704), new_size=(512, 512))
        resizeimage(InPath,OutPath4)
        displaylinesonimage(OutPath4, OutPath5, resized_left_imagelines, pt = 2)

        thisimage = {}
        thisimage["ID"] = f"{Number}_0"
        thisimage["2D_512"] = resized_left_imagelines
        thisimage["3D"] = resized_left_imagelines
        thisimage["ID_orig"] = [search1["ID"], search2["ID"]]

        allimage.append(thisimage)

    

    if len(right_imagelines) == 0:
        created_R = False
        print("NOLINES_R", Number, [search1["ID"], search2["ID"]])
        noimage = noimage + 1
    else:
        InPath =  TESTIMAGE + f"{Number}_1.png"
        OutPath1 =  TESTIMAGE_Lines + f"{Number}_1.png"
        OutPath2 = TESTIMAGE_GT + f"{Number}_1.png"
        OutPath3 = TESTIMAGE_GT_512 + f"{Number}_1.png"
        OutPath4 = TESTIMAGE_IMAGES_512 + f"{Number}_1.png"
        OutPath5 = TESTIMAGE_LINES_512 + f"{Number}_1.png"
        displaylinesonimage(InPath, OutPath1, right_imagelines, pt=4)
        displaylinesonimage_GT(OutPath2, right_imagelines)
        # 512
        resized_right_imagelines = resizedlinesGT(OutPath3, right_imagelines, old_size=(752, 704), new_size=(512, 512))
        resizeimage(InPath,OutPath4)
        displaylinesonimage(OutPath4, OutPath5, resized_right_imagelines, pt=2)

        thisimage = {}
        thisimage["ID"] = f"{Number}_1"
        thisimage["2D_512"] = resized_right_imagelines
        thisimage["3D"] = resized_right_imagelines
        thisimage["ID_orig"] = [search1["ID"], search2["ID"]]

        allimage.append(thisimage)


    #LEFT IMAGE
    

    
    
    return allimage,created_R, created_L,noimage





    







######### CODE STARTS #########

images_csv_data = read_csv(FileNameImage)
arrangearray(images_csv_data)

BlackList = []
allimage = []
noLineOnimage = 0
created_R = True
created_L = True
noimage = 0
Number = 0
while len(BlackList) < len(images_csv_data):
    print(len(BlackList),"----" ,len(images_csv_data))
    search1, BlackList , noLineOnimage= firstimage(BlackList, noLineOnimage)
    if len(search1) == 0:
        break
    search2, BlackList, noLineOnimage = second(BlackList, noLineOnimage, search1)
    if len(search2) == 0:
        break
    
    #image_paths = [folder_path + , folder_path + search2 + ".png"]
    # displaylinesonimage_search(folder_path, TESTIMAGE, search1)
    # displaylinesonimage_search(folder_path, TESTIMAGE, search2)
    stack_and_split_images(folder_path, TESTIMAGE, search1, search2, Number)

    allimage, created_R, created_L,noimage  = movelines(search1,search2, Number, TESTIMAGE, allimage, created_R, created_L,noimage )

    if (created_L == True or created_R == True):
        Number = Number + 1
        created_L = True 
        created_R = True 



fieldnames = ['ID', '2D_512', '3D', 'ID_orig']

print("NOIMAGE")
print(noimage)


write_data_to_csv(allimage, csvpath, fieldnames)
    



