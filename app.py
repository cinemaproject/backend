from flask import Flask, request, render_template
import insightface
import urllib
import urllib.request
import cv2
import json
import numpy as np

model = insightface.app.FaceAnalysis(rec_name = 'arcface_r100_v1', det_name = 'retinaface_mnet025_v2')
ctx_id = -1
model.prepare(ctx_id = ctx_id, nms=0.4)
app = Flask(__name__)

def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

@app.route('/')
def hello():
    return 'hello'


@app.route('/actors_recognition', methods=['GET'])
def recognize_actors():
    img_url = request.args.get('url')
    img = url_to_image(img_url)
    img_out = img.copy()
    faces = model.get(img)
    faces_arr = []
    for idx, face in enumerate(faces):
        face_d = {}
        #print("Face [%d]:"%idx)
        face_d['index'] = idx
        #print("\tage:%d"%(face.age))
        gender = 'Male'
        if face.gender==0:
            gender = 'Female'
        face_d['gender'] = gender
        #print("\tgender:%s"%(gender))
        #print("\tembedding shape:%s"%face.embedding.shape)
        face_d['embedding_shape'] = str(face.embedding.shape)
        bboxes = face.bbox.astype(np.int)
        face_d['bbox'] = str(bboxes.flatten())
        #print("\tbbox:%s"%(bboxes.flatten()))
        img_out = cv2.rectangle(img_out, (bboxes[0], bboxes[1]), (bboxes[2], bboxes[3]), (0, 255, 0), 2)
        #print("\tlandmark:%s"%(face.landmark.astype(np.int).flatten()))
        #print("")
        img_out = cv2.putText(img_out, str(face.age), (int(bboxes[0]-10), int(bboxes[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0),2)
        img_out = cv2.putText(img_out, str(gender), (int(bboxes[0]+40), int(bboxes[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255),2)
        img_out = cv2.putText(img_out, str(idx), (int(bboxes[0]-40), int(bboxes[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255),2)
        faces_arr.append(face_d)
    cv2.imwrite('static\\result.jpg', img_out) 
    resp = {"faces": faces_arr}
    #return render_template("actors.html", res_image = 'static\\result.jpg')
    return json.dumps(resp,ensure_ascii=False), 200


@app.route('/posters_by_color')
def hello_name():
    return "posters_by_color"


if __name__ == '__main__':
    app.run()
