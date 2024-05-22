import scipy.io
import csv
import ast
import numpy as np
from scipy.io import loadmat

#INPUT
# MatFile = "/Volumes/TIMKA/NewData/MATFILES/Data_img.mat"
MatFile = "/projnas/project/madjeda/MAIN_TRAINING_DATA_PERSPECTIVE/DataBase/DataBase_360/all_ID_SOLD_seq05/Data_img.mat"
#MatFile = "/project/ntimea/NewData/MATFILES/NewAugment/NewData/Trimmed_OPENCV.mat"

#OUTPU
#FileName = "/Volumes/TIMKA/NewData/MATFILES/Data_img_ORIG.csv"
FileName = "/project/ntimea/NewData/MATFILES/NewAugment/NewData/Augmented_OPENCV.csv"




input_mat = loadmat(MatFile)
osszesitett_data = input_mat["Data"]
#imageName, Lines2D, Lines3D, Lines3D_ID


allInputData = len(input_mat[0])
dicty = {}
alldata = []

for data in range(allInputData):
    print(data, "----", allInputData)

    imageID = (osszesitett_data["imageName"][0][data][0])
    
    image3D_seg = np.array(osszesitett_data["Lines3D"][0][data])
    image3D = []
    for id in image3D_seg:
        image3D.append(id.tolist())

    image2D_seg = np.array(osszesitett_data["Lines2D"][0][data])
    image2D = []
    for id in image2D_seg:
        image2D.append(id.tolist())


    image3DID_seg =  np.array(osszesitett_data["Lines3D_ID"][0][data])
    image3DID = []
    for id in image3DID_seg:
        image3DID.append(id[0][0])

    dicty = {}
    dicty["ID"] = imageID
    dicty["2D"] = image2D
    dicty["3D"] = image3D
    dicty["id_3D"] = image3DID

    alldata.append(dicty)

def write_data_to_csv(data_array, filename):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['ID', '2D', '3D', 'id_3D']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for data in data_array:
            writer.writerow(data)





write_data_to_csv(alldata, FileName)

 
    
    
    