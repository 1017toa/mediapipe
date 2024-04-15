# STEP 1
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
# from insightface.data import get_image as ins_get_image

# STEP 2
# app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app = FaceAnalysis(name='buffalo_l',providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# STEP 3
# img = ins_get_image('t1')
img = cv2.imread('images/q.jpg')
img2 = cv2.imread('images/qqq.jpg')

# STEP 4
faces = app.get(img)
faces2 = app.get(img2)

# STEP 5
rimg = app.draw_on(img, faces)
cv2.imshow("test",rimg)
cv2.waitKey(0)

# cv2.imwrite("./t1_output.jpg", rimg)

# then print all-to-all face similarity
# feats = []
# for face in faces:
#     feats.append(face.normed_embedding)
# feats = np.array(feats, dtype=np.float32)
# sims = np.dot(feats, feats.T)
# print(sims)

feat1 = np.array(faces[0].normed_embedding, dtype=np.float32)
feat2 = np.array(faces2[0].normed_embedding, dtype=np.float32)
sims = np.dot(feat1, feat2.T)
print(sims)