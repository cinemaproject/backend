#import config
from PIL import Image
import os
from flask_cors import CORS, cross_origin
import urllib
import urllib.request
import json
import numpy as np
import os
import cv2
import base64
import pandas as pd
from flask import Flask, request, render_template, send_file
from clipmodel import EmbedderOpenCLIP
from indexer import Indexer

def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

def readb64(uri):
   encoded_data = str(uri).split(',')[1]
   nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
   img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
   return img

df = pd.read_csv('merged.csv')
clip_embedder = EmbedderOpenCLIP(device = 'cpu', clip_model_name = 'ViT-B-16-plus-240', pretrained = 'laion400m_e32')

app = Flask(__name__)
indexer = Indexer()
indexer.load('merged.npy')
print(indexer.index.shape)
@app.route('/', methods=['GET'])
@cross_origin()
def helloworld():
    return 'hello'

@app.route('/search_by_image', methods=['POST'])
@cross_origin()
def search_by_image():
    content = request.json
    amount = content['amount']
    if content['type'] == 'base64':
        image_undecoded = content['image']
        img = readb64(image_undecoded)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif content['type'] == 'url':
        img = url_to_image(content['image'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    emb = clip_embedder.encode_imgs([img])
    dists, indexes = indexer.find(emb, topn = amount)
    loc = df.iloc[indexes]
    return {'dists': str(list(dists)), 'names' : list(loc.Name.values), 'urls': str(list(loc.Url.values)), 'db_size': str(len(df.index))}


@app.route('/search_by_prompt', methods=['POST'])
@cross_origin()
def search_by_prompt():
    content = request.json
    prompt = content['prompt']
    amount = content['amount']
    emb = clip_embedder.encode_text(prompt)
    dists, indexes = indexer.find(emb,topn = amount)
    loc = df.iloc[indexes]
    return {'dists': str(list(dists)), 'names' : list(loc.Name.values), 'urls': str(list(loc.Url.values)), 'db_size': str(len(df.index))}


if __name__ == '__main__':
    print("Starting server on port 8000")
    app.run(host="0.0.0.0", port=8000)
