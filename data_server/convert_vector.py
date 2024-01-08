import boto3
import numpy as np
from facenet_pytorch import MTCNN
import torch
import cv2
from pymongo import MongoClient
import torch 
from torchvision import transforms
import os
from PIL import Image
import numpy as np
from func.face_comparison.facenet import loadModel
from func.face_comparison.compare_faces import img_to_encoding
import torch
import numpy as np
import boto3
from arcface import ArcFace
import datetime
url = "mongodb+srv://vudinhtruongan:demo@test.6o5s3eu.mongodb.net/test"
client = MongoClient(url)
db = client['user_management1']
collection = db['users']
aws_access_key_id = 'AKIAUJUMLJ3YNYNZ5KU6'
aws_secret_access_key = '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'
s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
bucket_name = 'truongan912'
device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def l2_normalize(x):
    return x / np.sqrt(np.sum(np.multiply(x, x)))

def convert_vector(face_recognition,mtcnn,id,name): 
    response = s3.list_objects_v2(Bucket=bucket_name)
    print(response["Contents"])
    # model =  loadModel()
    os.makedirs("emb",exist_ok=True)

    today=datetime.date.today()
    # print("Today-------------: ",today)
    embeddings = []
    names = []
    embeds = []
    for obj in response['Contents']:
        if id==obj["Key"].split(".")[0]:
            key = obj['Key']

            # print("check key-------------------------",key)
            response = s3.get_object(Bucket=bucket_name, Key=key)
            image_data = response['Body'].read()
            image_data = np.frombuffer(image_data, np.uint8)
            image_data = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            print(image_data.shape)
            cv2.imwrite("o.jpg",image_data)
            boxes, _ = mtcnn.detect(image_data)
            face_img=None
            for box in boxes:
                box=list(map(int,box))
                face_img=image_data[box[1]:box[3],box[0]:box[2]]
  
            # print(face_img.shape)
            try:
                face=cv2.resize(face_img,(112,112))
                cv2.imwrite("test.jpg",face)
                # print(faccleae.shape)
                detect_embeds1 = face_recognition.calc_emb(face)
                print(detect_embeds1.shape)
                np.save(os.path.join("emb",name+".npy"),np.array(detect_embeds1))
                # detect_embeds1 = l2_normalize(detect_embeds1)
                detect_embeds_ = torch.tensor(detect_embeds1)
                print(detect_embeds_.shape)
                embeds.append(detect_embeds_.unsqueeze(0))
                embedding = torch.cat(embeds).mean(0, keepdim=True)
                embeddings.append(embedding)
                names.append(key)
            except:
                pass
    return embeddings, names
