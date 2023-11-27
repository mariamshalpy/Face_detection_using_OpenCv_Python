#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install opencv-python


# In[2]:


import cv2


# In[ ]:


#define a method to draw a box around the face
def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text):
    # Converting image to gray-scale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # detecting features in gray-scale image, returns coordinates, width and height of features
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []
    # drawing rectangle around the feature and labeling it
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)#2 is the thickness of the border
        cv2.putText(img, text, (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]
    return coords
    
    
    


# In[ ]:


# Method to detect the features
def detect(img, faceCascade, img_id):
    color = {"blue":(255,0,0), "red":(0,0,255), "green":(0,255,0), "white":(255,255,255)}
    coords = draw_boundary(img, faceCascade, 1.1, 10, color['blue'], "Face")
    # If feature is detected, the draw_boundary method will return the x,y coordinates and width and height of rectangle else the length of coords will be 0
    if len(coords)==4:
        # Updating region of interest by cropping image
        roi_img = img[coords[1]:coords[1]+coords[3], coords[0]:coords[0]+coords[2]]
        # Assign unique id to each user
        user_id = 1
        # img_id to make the name of each image unique
        generate_dataset(roi_img, user_id, img_id)

    return img


# In[ ]:


# Loading classifiers
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# In[3]:


#create object to the video capture and set the argument to 0 because we use the built-in webcam of our laptop or desktop.
video_capture=cv2.VideoCapture(0)


# In[ ]:


# Initialize img_id with 0
img_id = 0


# In[4]:


#create an infinte loop and we will read the video as an images 
while True:
     if img_id % 50 == 0:
        print("Collected ", img_id," images")
    # Reading image from video stream
    _, img = video_capture.read()
    # Call method we defined above
    img = detect(img, faceCascade, img_id)
    # Writing processed image in a new window
    cv2.imshow("Face Detection",img)
    img_id += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):#we will break the loop when the use press q
        break
   

# releasing web-cam
video_capture.release()
# Destroying output window
cv2.destroyAllWindows()
    


# In[ ]:




