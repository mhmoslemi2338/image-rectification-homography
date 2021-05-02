
import numpy as np
import matplotlib.pyplot as plt
import cv2
#import os
#pwd=os.getcwd()

def my_resize(im3,im4):
    h3=np.shape(im3)[0]
    (h4,w4)=np.shape(im4)[0:2]
    scale=h3/h4
    dim=(int(scale * w4) , h3)
    im4_resize=cv2.resize(im4.copy() , dim )
    return [im3,im4_resize]


###### reading images ######

im3_bgr=cv2.imread('im03.jpg',cv2.IMREAD_COLOR)
im3=cv2.cvtColor(im3_bgr,cv2.COLOR_BGR2GRAY)

im4_bgr=cv2.imread('im04.jpg',cv2.IMREAD_COLOR)
im4=cv2.cvtColor(im4_bgr,cv2.COLOR_BGR2GRAY)

[im4,im3]=my_resize(im4,im3)
[im4_bgr,im3_bgr]=my_resize(im4_bgr,im3_bgr)

#### find keypoint and descriptors ######

sift3=cv2.SIFT_create()
kp3, desc3 = sift3.detectAndCompute(im3, None)
sift4=cv2.SIFT_create()
kp4, desc4 = sift4.detectAndCompute(im4, None)

##### drawing keypoints on images ######
im3_kp=cv2.drawKeypoints(image=im3_bgr.copy(), keypoints=kp3 , outImage=None , color=(0,255,0))
im4_kp=cv2.drawKeypoints(image=im4_bgr.copy(), keypoints=kp4 , outImage=None , color=(0,255,0))

######  concatenate  and save two images #####
im_concat=cv2.hconcat([im3_kp,im4_kp])
cv2.imwrite('result/res13_corners.jpg',im_concat);


###################################
####### find matche points ########
###################################

matcher=cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_FLANNBASED)
thresh = 0.8

##### find matches for kp3 from image4 ####
match_pnt3=matcher.knnMatch(desc3,desc4,2)
valid_matche3 = []
for row in match_pnt3:
    a=row[0]
    b=row[1]
    dis_a=a.distance
    dis_b=b.distance
    if np.divide(dis_a,dis_b) < thresh :
        valid_matche3.append(a)
        
##### find matches for kp4 from image3 #####
match_pnt4=matcher.knnMatch(desc4,desc3,2)
valid_matche4 = []
for row in match_pnt4:
    a=row[0]
    b=row[1]
    dis_a=a.distance
    dis_b=b.distance
    if np.divide(dis_a,dis_b) < thresh :
        valid_matche4.append(a)    

##### find mutual corresponding point in both images (good matches) ####
valid=[]
for row3 in valid_matche3:
    k3=row3.queryIdx
    k4=row3.trainIdx
    for row4 in valid_matche4:
        k3_tmp=row4.trainIdx
        k4_tmp=row4.queryIdx
        if k3==k3_tmp:
            if k4==k4_tmp:
                valid.append([k3,k4])



######### draw match points ######
pnt_match3=[]
pnt_match4=[]

for row in valid:
    pnt_match3.append(kp3[row[0]])
    pnt_match4.append(kp4[row[1]])
    
im3_match=cv2.drawKeypoints(image=im3_kp.copy(), keypoints=pnt_match3 , outImage=None, color=(255,0,0))
im4_match=cv2.drawKeypoints(image=im4_kp.copy(), keypoints=pnt_match4 , outImage=None , color=(255,0,0))

######  concatenate  and save two images #####
im_concat2=cv2.hconcat([im3_match,im4_match])
cv2.imwrite('result/res14_correspondences.jpg',im_concat2);



###### drawing line for corespond points #####
(h3,w3)=np.shape(im3)[0:2]
im_concat3=im_concat2.copy()

for row in (valid): 
    pt3=tuple(map(int,(kp3[row[0]]).pt))
    (i4,j4)=tuple(map(int,(kp4[row[1]]).pt))
    cv2.line(im_concat3,pt3,(i4+w3,j4),color=(255,0,0),thickness=1)
    
cv2.imwrite('result/res15_matches.jpg',im_concat3);

##### drawing 20 line for corespond points #####
im_concat4=im_concat2.copy()
ch=0
for row in (valid): 
    if ch==20 : 
        break
    pt3=tuple(map(int,(kp3[row[0]]).pt))
    (i4,j4)=tuple(map(int,(kp4[row[1]]).pt))
    cv2.line(im_concat4,pt3,(i4+w3,j4),color=(255,0,0),thickness=1)
    ch+=1
    
cv2.imwrite('result/res16.jpg',im_concat4);


##### find homography with RANSAC #####

src_pt=np.array([row.pt for row in pnt_match4],dtype=np.float32)
dst_pt=np.array([row.pt for row in pnt_match3],dtype=np.float32)

homography, mask = cv2.findHomography(src_pt, dst_pt, cv2.RANSAC,60)
match_Mask = mask.tolist()
print(" Homography matrix is :\n" , homography)

#### draw inlier points ####

inlier3,inlier4=[],[]
for i in range(len(match_Mask)):
    if match_Mask[i][0]==1:
        [i1,i2]=valid[i]
        inlier3.append(pnt_match3[i])
        inlier4.append(pnt_match4[i])

im3_inlier=cv2.drawKeypoints(im3_match.copy(), inlier3 , None, (0,0,255))
im4_inlier=cv2.drawKeypoints(im4_match.copy(), inlier4 , None , (0,0,255))

im_concat5=cv2.hconcat([im3_inlier,im4_inlier])
 
#### draw lines for correspond points #### 
for i,row in enumerate(valid): 
    pt3=tuple(map(int,(kp3[row[0]]).pt))
    (i4,j4)=tuple(map(int,(kp4[row[1]]).pt))
    if match_Mask[i][0]==0:
        cv2.line(im_concat5,pt3,(i4+w3,j4),color=(255,0,0),thickness=1)
    else:
        cv2.line(im_concat5,pt3,(i4+w3,j4),color=(0,0,255),thickness=1)
        
cv2.imwrite('result/res17.jpg',im_concat5);


#### apply homography matrix to second image #####

im_final = cv2.warpPerspective(im4_bgr.copy(),homography,(3100,1400))
cv2.imwrite('result/res19.jpg',im_final);


