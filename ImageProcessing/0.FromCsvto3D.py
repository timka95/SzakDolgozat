

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
FilePath = "/project/ntimea/NewData/MATFILES/Data_img_SOLD_seq05.csv"

#OUTPUT
Filename = "/project/ntimea/NewData/MATFILES/NewAugment/NewData/3D_SOLD_seq05.csv"


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

dicty = {}
segedarray3did = []
bigarray = []

for i in range(len(csvdata)):

    print(i, "-----", len(csvdata))
    current3did = csvdata[i]["id_3D"]
    currentid = csvdata[i]["ID"]
    current2d = csvdata[i]["2D"]
    current3d = csvdata[i]["3D"]
    for j in range(len(current3did)):
        if current3did[j] in segedarray3did:
            index = segedarray3did.index(current3did[j])
            bigarray[index]["ID"].append(currentid)
            bigarray[index]["2D"].append(current2d[j])
        else:
            segedarray3did.append(current3did[j])
            dicty = {}
            dicty["id_3D"] = current3did[j]
            dicty["3D"] = current3d[j]
            dicty["2D"] = []
            dicty["2D"].append(current2d[j])
            dicty["ID"] = []
            dicty["ID"].append(currentid)
            bigarray.append(dicty)

    end = 0


def write_data_to_csv(data_array, filename):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['id_3D','3D', 'ID', '2D']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for data in data_array:
            writer.writerow(data)

write_data_to_csv(bigarray, Filename)


            


