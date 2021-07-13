import cv2
import numpy as np
from plyfile import PlyData,PlyElement
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from cross_junctions import cross_junctions
#================================================================================================================
#                                         PLY DATA PROCESS
#================================================================================================================
#start with data again
plydata = PlyData.read('new_cell2.ply')
#seperate the vertexs and faces from the ply file
vertex = plydata.elements[0]
face = plydata.elements[1]

(x,y,z) = (vertex[t] for t in('x','y','z'))
(R,G,B) = (vertex[t] for t in('red','green','blue'))
index = []
for i in range(vertex.count):
    if z[i] < -0.6 or z[i] > -0.4:
        index.append(i)

new_x = np.delete(x,index)
new_y = np.delete(y,index)
new_z = np.delete(z,index)
new_R = np.delete(R,index)
new_G = np.delete(G,index)
new_B = np.delete(B,index)

scale = 900
new_X = np.round((new_x - min(new_x))*scale).astype(int)
new_Y = np.round((new_y - min(new_y))*scale).astype(int)


rangeX =int(round((max(new_x) - min(new_x))*scale))+1
rangeY =int(round((max(new_y) - min(new_y))*scale))+1
new_img  = 255 * np.ones((rangeY,rangeX,3),dtype = np.uint8)
count = 0

print('the range of Y is:',rangeY,'the range of X is: ',rangeX)
for i in range(len(new_X)):
    if new_Y[i]>int(rangeY/5) and new_Y[i]<int(rangeY*4/5):
        if new_X[i]>int(rangeX/5) and new_X[i]<int(rangeX*4/5):
            new_img[new_Y[i]][new_X[i]][0] = new_R[i]
            new_img[new_Y[i]][new_X[i]][1] = new_G[i]
            new_img[new_Y[i]][new_X[i]][2] = new_B[i]
            count+=1
#plt.imshow(new_img,origin = 'lower');plt.show()



figure1 = plt.figure(figsize=(41,30))
from matplotlib.backends.backend_agg import FigureCanvasAgg
canvas = FigureCanvasAgg(figure1)
ColoUr = np.dstack((new_R,new_G,new_B))
ColoUr.resize((len(new_R),3))
ax = figure1.add_subplot(111)
ax.scatter(new_x,new_y,color = ColoUr/255,edgecolors =None)
ax.axis('off')
canvas.draw()
buf = canvas.buffer_rgba()
X= np.asarray(buf)
plt.imshow(X);plt.show()
#================================================================================================================
#                                         IMAGE LOAD AND FEATURE DETECT
#================================================================================================================
gray = cv2.cvtColor(new_img, cv2.COLOR_RGB2GRAY)#cv2.cvtColor(new_img, cv2.COLOR_RGB2GRAY)
#for i in range(len(new_X)):
#$    if gray[new_Y[i]][new_X[i]] > 70:
#        gray[new_Y[i]][new_X[i]] = int(255)
#plt.imshow(gray);plt.show()

blur = cv2.medianBlur(gray, 3)
# blur2 = cv2.medianBlur(X,3)
# gray2 = cv2.cvtColor(blur2,cv2.COLOR_RGB2GRAY)
#sharpen_kernel = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])#this detect edge

sharpen_kernel2 = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])
sharpen2 = cv2.filter2D(blur,-1,sharpen_kernel2)

#need to see the openning image and close image to see difference

#Tophat is the difference between input image and openning of the image

#sharpen_kernel = np.array([[0,-1,0], [-1,6,-1], [0,-1,0]])#sharpen
#sharpen = cv2.filter2D(sharpen2, -1, sharpen_kernel)
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
close = cv2.morphologyEx(sharpen2, cv2.MORPH_GRADIENT, kernel, iterations=3)

#thresh = cv2.adaptiveThreshold(sharpen, 50,cv2.ADAPTIVE_T
#
#
# , cv2.THRESH_BINARY, 3, 2)
cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
min_area = 5
max_area = 10000
image_number = 0
for c in cnts:
    area = cv2.contourArea(c)
    if area > min_area and area < max_area:
        x,y,w,h = cv2.boundingRect(c)
        #ROI = new_img[y:y+h, x:x+h]
        #cv2.imwrite('ROI_{}.png'.format(image_number), ROI)
        cv2.rectangle(new_img, (x, y), (x + w, y + h), (50,255,50), 1)
        image_number += 1

# plt.imshow(X);plt.show()
# plt.imshow(close);plt.show()
# #plt.imshow(thresh,origin = 'lower');plt.show()
# plt.imshow(sharpen);plt.show()
# plt.imshow(gray2);plt.show()
# plt.imshow(blur2);plt.show()

plt.imshow(new_img,origin = 'lower');plt.show()
plt.imshow(close,origin = 'lower');plt.show()
#plt.imshow(sharpen,origin = 'lower');plt.show()
plt.imshow(sharpen2,origin = 'lower');plt.show()
plt.imshow(blur,origin = 'lower');plt.show()
plt.imshow(gray,origin = 'lower');plt.show()

#boundaries
boundaries = np.zeros((2,4))
boundaries[1][0] = boundaries[1][1]= int(np.round(rangeY*(1/6)))
boundaries[1][3] = boundaries[1][2]= int(np.round(rangeY*(5/6)))
boundaries[0][0] = boundaries[0][3]= int(np.round(rangeX*(1/4)))
boundaries[0][1] = boundaries[0][2]= int(np.round(rangeX*(3/4)))
#Ipts = cross_junctions(new_img, boundaries)
"""
import cv2
import numpy as np
from plyfile import PlyData,PlyElement
import matplotlib.pyplot as plt
plydata = PlyData.read('anotherface5.ply')
#seperate the vertexs and faces from the ply file
vertex = plydata.elements[0]
face = plydata.elements[1]
(x,y,z) = (vertex[t] for t in('x','y','z'))
(R,G,B) = (vertex[t] for t in('red','green','blue'))
index = []
for i in range(vertex.count):
    if z[i] < -1:
        #R[i] = G[i] = B[i] = 255
        index.append(i)
new_x = np.delete(x,index)
new_y = np.delete(y,index)
new_z = np.delete(z,index)
new_R = np.delete(R,index)
new_G = np.delete(G,index)
new_B = np.delete(B,index)

new_X = np.round((new_x - min(new_x))*600).astype(int)
new_Y = np.round((new_y - min(new_y))*600).astype(int)


rangeX =int(round((max(new_x) - min(new_x))*600))+1
rangeY =int(round((max(new_y) - min(new_y))*600))+1
new_img  = 255 * np.ones((rangeY,rangeX,3),dtype = np.uint8)
count = 0
for i in range(len(new_X)):
    new_img[new_Y[i]][new_X[i]][0] = new_R[i]
    new_img[new_Y[i]][new_X[i]][1] = new_G[i]
    new_img[new_Y[i]][new_X[i]][2] = new_B[i]
    count+=1
plt.imshow(new_img,origin = 'lower');plt.show()
"""

def findindex(X_array,Y_array,x,y):
    indexY = np.where(Y_array == y)[0]
    indexX = np.where(X_array == x)[0]
    result = np.intersect1d(indexY,indexX)

    return result[0]



"""
#this is for extract matrix from the plot

figure1 = plt.figure(figsize=(30,20))
from matplotlib.backends.backend_agg import FigureCanvasAgg
canvas = FigureCanvasAgg(figure1)
ColoUr = np.dstack((new_R,new_G,new_B))
ColoUr.resize((len(new_R),3))
ax = figure1.add_subplot(111)
ax.scatter(new_x,new_y,color = ColoUr/255,edgecolors =None)
ax.axis('off')
canvas.draw()
buf = canvas.buffer_rgba()
X= np.asarray(buf)
plt.imshow(X)

plt.show()

"""

#finding the box
"""
gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
blur = cv2.medianBlur(gray, 3)
sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
sharpen = cv2.filter2D(gray, -1, sharpen_kernel)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
close = cv2.morphologyEx(sharpen, cv2.MORPH_TOPHAT, kernel, iterations=1)
cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
min_area = 500
max_area = 7000
image_number = 0
for c in cnts:
    area = cv2.contourArea(c)
    if area > min_area and area < max_area:
        x,y,w,h = cv2.boundingRect(c)
        ROI = new_img[y:y+h, x:x+h]
        cv2.imwrite('ROI_{}.png'.format(image_number), ROI)
        cv2.rectangle(new_img, (x, y), (x + w, y + h), (36,255,12), 2)
        image_number += 1
        
rho = 1  # distance resolution in pixels of the Hough grid
theta = np.pi / 180  # angular resolution in radians of the Hough grid
threshold = 15  # minimum number of votes (intersections in Hough grid cell)
min_line_length = 10  # minimum number of pixels making up a line
max_line_gap = 2  # maximum gap in pixels between connectable line segments
lines = cv2.HoughLinesP(close, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)
                        
for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(new_img,(x1,y1),(x2,y2),(255,0,0),1)
        
plt.imshow(new_img,origin = 'lower');plt.show()

#boundaries 
boundaries = np.zeros((2,4))
boundaries[1][0] = boundaries[1][1]= np.round(rangeY*(1/6)).astype(int)
boundaries[1][3] = boundaries[1][2]= np.round(rangeY*(5/6)).astype(int)
boundaries[0][0] = boundaries[0][3]= np.round(rangeX*(1/4)).astype(int)
boundaries[0][1] = boundaries[0][2]= np.round(rangeX*(3/4)).astype(int)
"""