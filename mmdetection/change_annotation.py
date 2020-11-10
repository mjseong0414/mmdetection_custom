import sys
import os
import pandas as pd
import mmcv

def func(x):
    if x == 5:
        return "fork_lift"
    elif x == 6:
        return "yard_chassis"
    elif x == 7:
        return "yard_truck"
    elif x == 8:
        return "truck_etc"
    elif x == 9:
        return "car"

def main():
    labels_path = os.path.abspath("/home/minjae/mjseong/mmdetection/data/krri/new_data/labels")
    labels_list = os.listdir(labels_path)
    
    class_names = [5,6,7,8,9]

    df = pd.DataFrame({"Image_ID":[], "Image_Width":[], "Image_Height":[], "Class_id":[], "XMin":[], "YMin":[], "Width":[], "Height":[], "Class_name":[]})
    lists = []
    for label_name in labels_list:
        image = mmcv.imread('./data/krri/new_data/images/' + label_name.rstrip(".txt")+".png")
        height, width = image.shape[:2] # image의 height와 width
        print(label_name)
        with open(labels_path + "/" + label_name, "r") as ann:
            image_id = label_name[:-4] + ".png"
            file = ann.readlines()
            for i in file:
                i = i[:-3]
                i = i.split(" ")
                lists.append(image_id)
                lists.append(int(width))
                lists.append(int(height))
                lists.append(int(i[0]))
                lists.append(round((float(i[1]) - float(i[3])/2) * width, 2)) # xmin
                lists.append(round((float(i[2]) - float(i[4])/2) * height, 2)) # ymin
                lists.append(round(float(i[3]), 2)) # width
                lists.append(round(float(i[4]), 2)) # height
                a = pd.DataFrame(data=[lists], columns=["Image_ID","Image_Width","Image_Height","Class_id","XMin", "YMin", "Width", "Height"])
                df = df.append(a)
                df = df.reset_index(drop=True)
                del lists[:]

    df = df.astype({'Class_id':int})
    df = df.astype({'Image_Width':int})
    df = df.astype({'Image_Height':int})            
    df = df[df['Class_id'].isin(class_names)]    
    # df["Class_name"] = ["Fork_Lift" if c_id == 5 elif c_id == 6 for c_id in df["Class_id"]]
    df["Class_name"] = df["Class_id"].apply(lambda x: func(x))
    # df.to_csv("krri_annotations.csv", mode='w')
    df.to_csv("krri_annotations02.csv", index=False)

if __name__ == '__main__':
    main()