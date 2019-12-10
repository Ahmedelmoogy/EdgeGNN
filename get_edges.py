import numpy as np
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm





orb = cv2.ORB_create()

sift = cv2.xfeatures2d.SURF_create()

freak = cv2.xfeatures2d.FREAK_create()

sift.setUpright(True)


#print sift.getUpright()
features2 = []
des_list = []
kp_list = []
imgs = []
imgs_depth = []

with open('all') as f:
   for line in f:
     imgs.append(line)


images_list  = imgs

with open('all_depth') as f:
   for line in f:
     imgs_depth.append(line)




matches = []


features_all = []
def extract_edges(images,min_th,max_th):
    e = []
    for i in tqdm(range(len(images))):
      img1 = cv2.imread(images[i].strip('\n'),1)
      edges1 = cv2.Canny(img1,min_th,max_th)
      indices = np.where(edges1 != [0])
      indices = np.array(indices)
      #print np.max(indices[0])
      ##print np.max(indices[1])
      print np.transpose(indices).shape
      coordinates = zip(indices[0], indices[1])
      rgb_values = img1[indices[0,:],indices[1,:],:]
      features = np.concatenate((np.transpose(indices),rgb_values),axis = 1)
      print features.shape
      features_all.append(features)
      #print coordinates
      #print rgb_values.shape
      '''
      plt.subplot(121),plt.imshow(img1,cmap = 'gray')
      plt.title('Original Image'), plt.xticks([]), plt.yticks([])
      plt.subplot(122),plt.imshow(edges1,cmap = 'gray')
      plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
      plt.show()
      '''
      #print edges1
      e.append(edges1)
    edges = np.asarray(e)
    return edges





features_all = np.asarray(features_all)


print features_all.shape
edges= extract_edges(images_list,500,600)
print edges.shape
#print dst_points.shape

np.save('edges_all',edges)
#np.save('heads_dst_pts_100',dst_points)

'''

print src_points[0]
focal_length = 585

center = (320,240)
camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double")






dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion


print (depth_dst_points[0])




trans = []
rot = []

for i in tqdm(range(len(src_points))):

    d = np.array(depth_dst_points[i],dtype = 'float')
    _,rvecs, tvecs, inliers = cv2.solvePnPRansac(d,dst_points[i], camera_matrix, dist_coeffs ,flags = cv2.SOLVEPNP_ITERATIVE)
    trans.append(tvecs)
    rot.append(rvecs)







trans = np.asarray(trans).reshape((len(trans),3))



x = np.loadtxt('seq1.txt')[:,0:3]

print trans.shape
print x.shape

pre = [x[0]]


for i in range(len(trans)-1):
   print '--------' 
   print trans[i] + np.asarray(pre[-1])
   pre.append((trans[i]) + np.asarray(pre[-1]))

pre = np.asarray(pre)
print pre
plt.plot(pre[:,0],pre[:,1],label = 'Estimated')
plt.plot(x[:100,0],x[:100,1],label = 'Ground Truth')
plt.legend()
plt.show()


                      

'''

   


   
     

