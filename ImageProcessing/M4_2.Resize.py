
import csv
import ast
import cv2
import numpy as np 

# INPUT
mozaikarraypath = '/project/ntimea/NewData/MATFILES/NewAugment/NewData/Mozaik4_OPENCV.csv'

#OTPUT
#GT_square_PATH = "/project/ntimea/NewData/IMAGES/stack/NewCut_ALL2_square_1408/"
GT_512_PATH = "/project/ntimea/NewData/IMAGES/ImageProcessing/Mozaik4/GT_OPENCV/"
mozaikarraypath_ALL = '/project/ntimea/NewData/MATFILES/NewAugment/NewData/Mozaik4_OPENCV.csv'

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
    

def displaylines(outputpath, allines, old_size, new_size):
    
    newallines = []
    newlines = []

    img = np.zeros(new_size, dtype=np.uint8)

    # Calculate the scaling factor
    scale_factor = new_size[0] / old_size[0]

    for lines in allines:
        newlines = []
        for line in lines:
            # Scale the coordinates
            x1 = int(line[0] * scale_factor)
            y1 = int(line[1] * scale_factor)
            x2 = int(line[2] * scale_factor)
            y2 = int(line[3] * scale_factor)

            # Draw the line on the resized image
            img = cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 2)  
            newlines.append([x1,y1,x2,y2])
        newallines.append(newlines)

    #print(outputpath)
    cv2.imwrite(outputpath, img)

    return newallines


def write_data_to_csv(data_array, filename, fieldnames):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for data in data_array:
            writer.writerow(data)




mozaikdata = read_csv(mozaikarraypath)

for i in range(len(mozaikdata)):
    mozaikdata[i] = mozaikdata[i][0] 
    # NewCutted3DID[i]["ID"] = ast.literal_eval(NewCutted3DID[i]["ID"])
    mozaikdata[i]["ID"] = ast.literal_eval(mozaikdata[i]["ID"])
    mozaikdata[i]["2D"] = ast.literal_eval(mozaikdata[i]["2D"])
    mozaikdata[i]["3D"] = ast.literal_eval(mozaikdata[i]["3D"])
    mozaikdata[i]["id_3D"] = ast.literal_eval(mozaikdata[i]["id_3D"])
    mozaikdata[i]["ID_orig"] = ast.literal_eval(mozaikdata[i]["ID_orig"])
    mozaikdata[i]["2D_orig"] = ast.literal_eval(mozaikdata[i]["2D_orig"])


allmozaikarray = []
number = 0

for data in mozaikdata:
    number = number+1
    print(number)
    # print(data["2D"])
    i#magename = GT_square_PATH + str(data["ID"]) + ".png"
    #data_2D1408 = displaylines(imagename, data["2D"], old_size=(1504, 1408), new_size=(1408, 1408))
    imagename = GT_512_PATH + str(data["ID"]) + ".png"
    data_2D512 = displaylines(imagename, data["2D"], old_size=(1504, 1408), new_size=(512, 512))

    mozaikdictionary = {}
    mozaikdictionary["ID"] = data["ID"]
    mozaikdictionary["2D"] = data["2D"]
    #mozaikdictionary["2D_1408"] = data["ID"]
    #mozaikdictionary["2D_1408"] = data_2D1408
    mozaikdictionary["2D_512"] = data_2D512
    mozaikdictionary["3D"] = data["3D"]
    mozaikdictionary["id_3D"] = data["id_3D"]
    mozaikdictionary["ID_orig"] = data["ID_orig"]
    mozaikdictionary["2D_orig"] = data["2D_orig"]


    allmozaikarray.append(mozaikdictionary)

fieldnames = ['ID','2D', '2D_512', '2D_1408', '3D', 'id_3D', 'ID_orig' ,"2D_orig"]

write_data_to_csv(allmozaikarray, mozaikarraypath_ALL, fieldnames)
