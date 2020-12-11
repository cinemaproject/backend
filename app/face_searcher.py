import pickle
import json
import numpy as np
from scipy.spatial import distance

class face_searcher():
    
    def __init__(self):
        with open('embeddings.pickle', 'rb') as handle:
            self.d = pickle.load(handle)
        self.list_items = []
        for item in list(self.d.values()):
            self.list_items.append(item.squeeze())
        self.keys = []
        raw_keys = json.load( open( "keys_dict2.json" ) )
        for key, value in raw_keys.items():
            self.keys.append({"id": key, "image": value})
    
    
    def search_face(self, eval_emb, k):
    # Query dataset, k - number of closest elements (returns 2 numpy arrays)
        return_d = {}
        q_labels = distance.cdist(np.expand_dims(eval_emb, axis=0), np.array(self.list_items), 'cosine')
        argmx = np.argsort(q_labels[0])[:k]
        for i in range(len(argmx)):
            return_d['index'] = i
            return_d['dist'] = q_labels[0][argmx[i]]
            return_d['image_url'] = self.keys[int(argmx[i])]['image']
            return_d['id'] = self.keys[int(argmx[i])]['id']
        return return_d
