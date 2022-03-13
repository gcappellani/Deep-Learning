import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import OneHotEncoder


class Face:
    def __init__(self, filename, height, width, flat_data, data,
                 user_id, head_position, expression, eye_state, img_scale):
        self.filename = filename
        self.height = height
        self.width = width
        self.flat_data = flat_data
        self.data = data
        self.user_id = user_id
        self.head_position = head_position
        self.expression = expression
        self.eye_state = eye_state
        self.img_scale = img_scale


def readpgm(filename):
    pgm = Image.open(filename)

    flat_data = list(pgm.getdata())
    data = []
    for i in range(pgm.height) :
        data.append(flat_data[i * pgm.width : (i+1) * pgm.width])

    return pgm.height, pgm.width, flat_data, data


def getFace(fname):
    try:
        attr = fname[:-4].split('_')
        if len(attr) == 4: attr.append('1')
        pgm = readpgm(fname)
        return Face(fname, pgm[0], pgm[1], pgm[2], pgm[3],
                          attr[0], attr[1], attr[2], attr[3], attr[4])
    except Exception:
        pass


def traverse_dir():
    rootDir = 'faces'

    faces = []
    for dirName, subdirList, fileList in os.walk(rootDir) :
        for fname in fileList :
            face = getFace(dirName + '/' + fname)
            if face is not None :
                faces.append(face)

    return faces


class FaceEncoder:
    def __init__(self):
        self.encoder = OneHotEncoder()


    def get_working_set(self, img_scale, user_id=False, head_position=False, expression=False, eye_state=False) :
        faces = traverse_dir()

        X = []
        Y = []
        for face in faces:
            if face.img_scale == img_scale :
                X.append(face.flat_data)

                temp = []
                if user_id : temp.append(face.user_id)
                if head_position : temp.append(face.head_position)
                if expression : temp.append(face.expression)
                if eye_state : temp.append(face.eye_state)
                Y.append(temp)

        self.encoder.fit(Y)
        Y = self.encoder.transform(Y)

        return np.asarray([[input/155 for input in inputs] for inputs in X]), Y.toarray()


    def getHypothesis(self, Y):
        temp = [len(self.encoder.categories_[i]) for i in range(len(self.encoder.categories_))]
        out_len = sum(temp)
        pattern = [0]
        for i in range(len(temp)):
            pattern.append(pattern[i] + temp[i])

        hyp = []
        for y in Y:
            temp = [.0] * out_len
            for c in range(len(pattern) - 1):
                m = max(y[pattern[c]:pattern[c + 1]])
                indx = list(y[pattern[c]:pattern[c + 1]]).index(m)
                temp[pattern[c] + indx] = 1.0
            hyp.append(temp)

        return hyp


    def decode(self, Y):
        return self.encoder.inverse_transform(Y)