from scipy.io import savemat
import scipy.io
import numpy as np
from PIL import Image, ImageDraw
import math
import matplotlib.pyplot as plt
import scipy.io as sio
import csv
import ast

#INPUT
#data = scipy.io.loadmat('/Users/timeanemet/Desktop/CNN/matfiles/subset_data.mat')
FilePath = "/project/ntimea/NewData/MATFILES/Data_img_SOLD_seq05.csv"
#OUTPUT
#filename = '/project/ntimea/NewData/MATFILES/NewAugment/NewData/Cutted_OPENCV.csv'
filenameTEST = '/project/ntimea/NewData/MATFILES/NewAugment/NewData/Cutted_Data_img_SOLD_seq05.csv'

def read_csv(file_path):
    # Read the CSV file
    with open(file_path, 'r') as csvfile:
        # Create a CSV reader object
        csvreader = csv.reader(csvfile)
        # Get the header
        header = next(csvreader)
        # Create an empty list to store the csvdata[i]
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

csvdata = read_csv(FilePath)

for i in range(len(csvdata)):
    csvdata[i] = csvdata[i][0]
    csvdata[i]["ID"] = csvdata[i]["ID"]
    csvdata[i]["id_3D"] = ast.literal_eval(csvdata[i]["id_3D"])
    csvdata[i]["2D"] = ast.literal_eval(csvdata[i]["2D"])
    csvdata[i]["3D"] = ast.literal_eval(csvdata[i]["3D"])




# data_to_save = []
# imglines = []
# img3dlines = []
data = {}
dataarray = []
# lines_2D = []
# lines_3D = []
newdataarray = []
# maxcuts = 0
# maximage = ""

for i in range(len(csvdata)):

    print(i, "----", len(csvdata))

    data["ID"] = csvdata[i]["ID"]
    data["2D"] = csvdata[i]["2D"]
    data["3D"] = csvdata[i]["3D"]
    data["id_3D"] = csvdata[i]["id_3D"]


    
    dataarray.append(data)
    lines = data["2D"]


    # Converting the list of lists to a list of dictionaries
    lines_with_keys = [{'x1': line[0], 'y1': line[1], 'x2': line[2], 'y2': line[3]} for line in lines]

    # Make them in order
    for lines in lines_with_keys:
        if(lines["x2"] < lines["x1"]):
            seged = lines["x1"]
            lines["x1"] = lines["x2"]
            lines["x2"] = seged

            seged = lines["y1"]
            lines["y1"] = lines["y2"]
            lines["y2"] = seged


    # Sort based on 'x1' value
    lines_with_keys = sorted(lines_with_keys, key=lambda line: line['x1'])

    smallimgsize=376
    img_width = 1408
    img_height = 376

    



    for line in lines_with_keys:
        line["x1"] = math.floor(line["x1"])
        line["x2"] = math.ceil(line["x2"])

        if(line["y1"] < line["y2"]):
            line["y1"] = math.floor(line["y1"])
            line["y2"] = math.ceil(line["y2"])
        else:
            line["y2"] = math.floor(line["y2"])
            line["y1"] = math.ceil(line["y1"])



    onesarray = [0] * img_width

    for line in lines_with_keys:
        if line["x2"] > 1408:
            line["x2"] = 1408
        for j in range(line["x1"], line["x2"]):
            onesarray[j] = onesarray[j] + 1

    linesegarray = []
    linesegments = {}

    
    

    k = 0
    while k < len(onesarray):
        if onesarray[k] != 0:
            linesegments = {"x1": k}
            while k < len(onesarray) and onesarray[k] != 0:
                k += 1
            linesegments["x2"] = k - 1  # Subtract 1 to get the endpoint
            linesegarray.append(linesegments)
        else:
            k += 1

    cutpoints = {}
    cutpointsarray = []

    


    

    for line in linesegarray:
        segmentlength = line["x2"] - line["x1"]
        if(segmentlength < 376):
            middle = line["x1"] + math.ceil(segmentlength/2)
            cutpoints = {"start": middle - 188, "end": middle + 188}

            if(cutpoints["start"] < 0):
                cutpoints["start"] = 0
                cutpoints["end"] = 376

            if(cutpoints["end"] > 1408):
                cutpoints["end"] = 1408
                cutpoints["start"] = 1408-376
            cutpointsarray.append(cutpoints)
        else:
            cutpoints = {"start": line["x1"], "end": line["x1"]+376}
            cutpointsarray.append(cutpoints)
            cutpoints = {"start": line["x2"] - 376, "end": line["x2"]}

            if(cutpoints["start"] < 0):
                cutpoints["start"] = 0
                cutpoints["end"] = 376

            if(cutpoints["end"] > 1408):
                cutpoints["end"] = 1408
                cutpoints["start"] = 1408-376

            cutpointsarray.append(cutpoints)


    arrayseg = [0] * 2
    savearray = []

    for lines in cutpointsarray:
        
        arrayseg[0] = lines["start"]
        arrayseg[1] = lines["end"]
        savearray.append(arrayseg)
        arrayseg = [0] * 2 # Reset arrayseg for the next iteration

    

    # if(maxcuts < len(savearray)):
    #     maxcuts = len(savearray)
    #     maximage = image_id_1

    list = dataarray[i]

    
    segedarray = dataarray[i].copy()
    linesinimage_2D = []
    linesinimage_3D = []
    linesinimage_3DID = []
    linesinimage_2D_original = []
    newdata = {}
    numberofimages = 1


    LINE_2D = segedarray["2D"]

    for cuts in savearray:
        
        start = cuts[0]
        end = cuts[1]
        for lines in LINE_2D[:]:
            x1 = lines[0]
            y1 = lines[1]
            x2 = lines[2]
            y2 = lines[3]

            if(x1 < end):
                x1 = x1-start
                x2 = x2-start
                newline = (x1,y1,x2,y2)
                linein3D = segedarray["3D"][segedarray["2D"].index(lines)]
                linein3DID = segedarray["id_3D"][segedarray["2D"].index(lines)]
                linesinimage_3D.append(linein3D)
                linesinimage_3DID.append(linein3DID)
                linesinimage_2D.append(newline)
                linesinimage_2D_original.append(lines)
                if(x2<end-start):
                    LINE_2D.remove(lines)
                    segedarray["3D"].remove(linein3D)
                
                
        if linesinimage_2D != []:
            newname = data["ID"] + "_" + str(numberofimages)
            newdata["ID"] = newname
            newdata["2D"] = (linesinimage_2D)
            newdata["2D_orig"] = (linesinimage_2D_original)
            newdata["3D"] = (linesinimage_3D)
            newdata["id_3D"] = (linesinimage_3DID)
            #print(linesinimage_3D)
            newdata["cutedhere"] = (cuts)
            newdataarray.append(newdata)
            newdata = {}
            linesinimage_2D = []
            linesinimage_3D = []
            linesinimage_2D_original = []
            numberofimages = numberofimages + 1



# # Initialize an empty dictionary to hold all the data
# output_data = {
#     'data2': newdataarray
# }

# scipy.io.savemat('/Users/timeanemet/Desktop/CNN/matfiles/data_pairs_cutted2.mat', output_data,  long_field_names=True)


def write_data_to_csv(data_array, filename):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['ID', '2D', '2D_orig', '3D', 'id_3D', 'cutedhere']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for data in data_array:
            writer.writerow(data)

write_data_to_csv(newdataarray, filenameTEST)

