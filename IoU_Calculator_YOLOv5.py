import pandas as pd
import cv2
import numpy as np
from shapely.geometry import Polygon

def calculate_iou(box_1, box_2):
    poly_1 = Polygon(box_1)
    poly_2 = Polygon(box_2)
    iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
    return iou


def load_bounding_box(lokasigambar,lokasitxt1,lokasitxt2):
    
    df = pd.read_csv(lokasitxt1, sep=" ", header=None,names=["Label", "x1", "y1", "xw", "yw"])
    df1 = pd.read_csv(lokasitxt2, sep=" ", header=None,names=["Label", "x1", "y1", "xw", "yw"])


    #read image
    img_path  = lokasigambar
    cvmat = cv2.imread(img_path)

    df_dset = pd.DataFrame()

    dot1_1=[]
    dot2_1=[]
    dot3_1=[]
    dot4_1=[]
    dot1_2=[]
    dot2_2=[]
    dot3_2=[]
    dot4_2=[]



    #get height, width

    h,w,_ = cvmat.shape

    #expert label
    for index, row in df.iterrows():
        #extract x1, y1 <- center, width, height

        x1 = int( row['x1'] * w )

        y1 = int( row['y1'] * h )

        xw = int( row['xw'] * w /2)

        yw = int( row['yw'] * h /2)

        #make x1,y1, x2,y2

        titik1_1 = [x1 - xw, y1 - yw ]
        titik2_1   = [x1 + xw, y1 - yw ]
        titik3_1   = [x1 + xw, y1 + yw ]
        titik4_1   = [x1 - xw, y1 + yw ]
        
        dot1_1.append(titik1_1)
        dot2_1.append(titik2_1)
        dot3_1.append(titik3_1)
        dot4_1.append(titik4_1)
    
    #yolov5 label prediction
    for index, row in df1.iterrows():
        #extract x1, y1 <- center, width, height

        x1_1 = int( row['x1'] * w )

        y1_1 = int( row['y1'] * h )

        xw_1 = int( row['xw'] * w /2)

        yw_1 = int( row['yw'] * h /2)

        #make x1,y1, x2,y2

        titik1_2 = [x1_1 - xw_1, y1_1 - yw_1 ]
        titik2_2   = [x1_1 + xw_1, y1_1 - yw_1 ]
        titik3_2   = [x1_1 + xw_1, y1_1 + yw_1 ]
        titik4_2   = [x1_1 - xw_1, y1_1 + yw_1 ]
        
        dot1_2.append(titik1_2)
        dot2_2.append(titik2_2)
        dot3_2.append(titik3_2)
        dot4_2.append(titik4_2)

    df_dset['dot1_expert'] = dot1_1
    df_dset['dot2_expert'] = dot2_1
    df_dset['dot3_expert'] = dot3_1
    df_dset['dot4_expert'] = dot4_1
    df_dset['dot1_yolov5'] = dot1_2
    df_dset['dot2_yolov5'] = dot2_2
    df_dset['dot3_yolov5'] = dot3_2
    df_dset['dot4_yolov5'] = dot4_2

    max_valuepd = pd.DataFrame()

    # kali silang dari semua bbx
    for i in df_dset.index:
            max_value =[]
            for i1 in df_dset.index:
                box_1 = [df_dset['dot1_expert'][i], df_dset['dot2_expert'][i], df_dset['dot3_expert'][i], df_dset['dot4_expert'][i]]
                box_2 = [df_dset['dot1_yolov5'][i1], df_dset['dot2_yolov5'][i1], df_dset['dot3_yolov5'][i1], df_dset['dot4_yolov5'][i1]]
                akhir = calculate_iou(box_1, box_2)
                max_value.append(akhir)
            max_valuepd[i] = max_value 
    
    iou = []
    
    #max bbx of max_valuepd
    for i in max_valuepd.columns:
        #print(i)
        
        iou.append(max_valuepd.iloc[i-1].max())
    final_iou = pd.DataFrame(iou)
    return final_iou


image_file_location = "C/yourlocation/file"
labeltxt_Groundtruth_file_location = "C/yourlocation/file"
labeltxt_Predictions_file_location = "C/yourlocation/file"


df_dset = load_bounding_box()




