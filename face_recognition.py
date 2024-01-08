from func.lib import *
# from mtcnn import MTCNN
from facenet_pytorch import MTCNN
from pymongo import MongoClient
import tensorflow as tf
from data_server.convert_vector import convert_vector
from arcface import ArcFace

# ip cam - modify for each cam
ip_address = "192.168.10.108"
port = 5542
username = "admin"
password = "Neon1234"

# URL for the camera stream
stream_url = f"rtsp://{username}:{password}@{ip_address}:{port}//Streaming/Channels/2"

# mongodb
url = "mongodb+srv://vudinhtruongan:demo@test.6o5s3eu.mongodb.net/test"
# client = MongoClient(url)
# db = client['user_management1']
# collection = db['users']

# local
frame_size = (640,480)
IMG_PATH = 'data_server/data'
DATA_PATH = 'data_server/path'
 
# Sử dụng ArcFace 
face_recognition=ArcFace.ArcFace()

def l2_normalize(x):
    return x / np.sqrt(np.sum(np.multiply(x, x)))


def trans(img):
    transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    return transform(img).unsqueeze(0)


def load_faceslist():
    if device == 'cpu':
        embeds = torch.load(DATA_PATH+'/faceslistCPU.pth')
    else:
        embeds = torch.load(DATA_PATH+'/faceslist.pth')
    names = np.load(DATA_PATH+'/usernames.npy')
    return embeds, names





def inference(face1, local_embeds, threshold = 3):
    face = np.array(face1)
    detect_embeds1 = face_recognition.calc_emb(face)
    # detect_embeds1 = l2_normalize(detect_embeds1)
    detect_embeds = torch.tensor(detect_embeds1)
    
                    #[1,512,1]                                      [1,512,n]
    norm_diff = detect_embeds.unsqueeze(-1) - torch.transpose(local_embeds, 0, 1).unsqueeze(0)
    # print(norm_diff)
    norm_score = torch.sum(torch.pow(norm_diff, 2), dim=1) #(1,n), moi cot la tong khoang cach euclide so vs embed moi

    min_dist, embed_idx = torch.min(norm_score, dim = 1)
   
    if min_dist > 0.3:
        return -1, -1
    else:
        return embed_idx, min_dist.double()

def extract_face(box, img, margin=20):
    face_size = 112
    img_size = frame_size
    margin = [
        margin * (box[2] - box[0]) / (face_size - margin),
        margin * (box[3] - box[1]) / (face_size - margin),
    ] #tạo margin bao quanh box cũ
    img = img[box[1]:box[3], box[0]:box[2]]
    # img = img[box[1]:box[1]+box[3], box[0]:box[2]+box[0]]
    face = cv2.resize(img,(face_size, face_size), interpolation=cv2.INTER_AREA)
    # face = Image.fromarray(face)
    return face




if __name__ == "__main__":
    prev_frame_time = 0
    new_frame_time = 0
    power = pow(10, 6)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # model = loadModel()
    people_list = {}

    cap = cv2.VideoCapture(0) # 0 for webcam, 3 for out_cam, stream_url for ip_cam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    
    while cap.isOpened():
        isSuccess, frame = cap.read()
        # embeddings, names = load_face_vector()
        if isSuccess:
            
# face detectiom

    # mtcnn
            mtcnn = MTCNN(thresholds= [0.6, 0.7, 0.7] ,keep_all=True, device = device)
            boxes, _ = mtcnn.detect(frame)

    # # mtcnn new ver
    #         tf_mtcnn = MTCNN()
    #         resp = tf_mtcnn.detect_faces(frame)
    #         boxes_ = np.array([item['box'] for item in resp])
    #         boxes = boxes_.copy()  # Create a copy to avoid modifying the original matrix
    #         boxes[:, 2] += boxes_[:, 0]
    #         boxes[:, 3] += boxes_[:, 1]

    # retinaface   
            # resp = RetinaFace.detect_faces(frame)
            # facial_areas = []
            # try:
            #     for face_key, face_value in resp.items():
            #         facial_area = face_value['box']
            #         facial_areas.append(facial_area)
            #     boxes = np.atleast_2d(facial_area)
            # except:
            #     boxes = None

            

            try:
                for box in boxes:
                    bbox = list(map(int,box.tolist()))
                    face = extract_face(bbox, frame)
                    emb_vector=face_recognition.calc_emb(face)
                    MIN=99
                    name=None
                    for vector in os.listdir("emb"):
                        data=np.load(os.path.join("emb",vector))
                        dist=face_recognition.get_distance_embeddings(np.array(emb_vector),data)

                        if dist<MIN:
                            MIN=dist
                            name=vector.split(".")[0]
                    print(MIN)
                    if MIN<0.5:
                        frame = cv2.rectangle(frame, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (0,0,255), 6)
                        # score = torch.Tensor.cpu(score[0]).detach().numpy()
                        frame = cv2.putText(frame, name, (bbox[0],bbox[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,0), 2, cv2.LINE_8)
                        # print(names[idx] + '_{:.2f}'.format(score))

                        # people_list[str(datetime.datetime.now())]=str(names[idx])
                        
                        
                    else:
                        frame = cv2.rectangle(frame, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (0,0,255), 6)
                        frame = cv2.putText(frame,'Unknown', (bbox[0],bbox[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,0), 2, cv2.LINE_8)
                        print("Unknown")

                new_frame_time = time.time()
                fps = 1/(new_frame_time-prev_frame_time)
                prev_frame_time = new_frame_time
                fps = str(int(fps))
                cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_DUPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
            except:
                pass

        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1)&0xFF == ord("q"):
            break


        current_datetime = datetime.datetime.now()

        # Generate a formatted string with the date and time
        formatted_datetime = current_datetime.strftime("%d-%m-%Y")

        # Define the file name with the formatted datetime
        pp_list = f"./processing_data/data/{formatted_datetime}.json"


        # Open the file with the generated file name
        with open(pp_list, 'w') as f:
            # Write the dictionary to the file in JSON format
            json.dump(people_list, f)



    cap.release()
    cv2.destroyAllWindows()
    #+": {:.2f}".format(score)

    