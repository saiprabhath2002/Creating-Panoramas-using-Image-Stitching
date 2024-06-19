import numpy as np
import cv2
import math
import os
import random
import sys
import shutil


def gaus_of_discriptor(size, sigma,center):
    # center = (size - 1) / 2
    x, y = np.meshgrid(np.arange(0, size), np.arange(0, size))
    exponent = -((x - center[1])**2 + (y - center[0])**2) / (2 * sigma**2)
    gaussian_values = np.exp(exponent)
    return gaussian_values / np.sum(gaussian_values)

def gaussian_matrix(size, sigma):
    center = (size - 1) / 2
    x, y = np.meshgrid(np.arange(0, size), np.arange(0, size))
    exponent = -((x - center)**2 + (y - center)**2) / (2 * sigma**2)
    gaussian_values = np.exp(exponent)
    return gaussian_values / np.sum(gaussian_values)


def generate_convolution_windows(image, kernel,other=True):
    padded_image = np.pad(image, ((kernel.shape[0] // 2, kernel.shape[0] // 2), (kernel.shape[1] // 2, kernel.shape[1] // 2)), mode='constant',constant_values=255)
    num_windows = (padded_image.shape[0] - kernel.shape[0] + 1) * (padded_image.shape[1] - kernel.shape[1] + 1)
    windows_matrix = np.lib.stride_tricks.sliding_window_view(padded_image, kernel.shape).reshape(num_windows, -1)
    return windows_matrix


def convolve(image, kernel):
    windows_matrix = generate_convolution_windows(image, kernel)
    flattened_kernel = kernel.flatten()
    result = np.dot(windows_matrix, flattened_kernel)
    result_height = image.shape[0]+2*(kernel.shape[0] // 2) - kernel.shape[0] + 1
    result_width = image.shape[1]+2*(kernel.shape[1] // 2) - kernel.shape[1] + 1
    result = result.reshape(result_height, result_width)
    return result


def generate_convolution_windows2(image, kernel):
    pad_height = kernel.shape[0] // 2
    pad_width = kernel.shape[1] // 2
    num_windows = (image.shape[0] - kernel.shape[0] + 1) * (image.shape[1] - kernel.shape[1] + 1)
    all_windows = np.lib.stride_tricks.sliding_window_view(image, kernel.shape)
    windows_matrix = all_windows.reshape(num_windows, -1)

    return windows_matrix


def convolve_withoutpadding(image, kernel):
    windows_matrix = generate_convolution_windows2(image, kernel)
    flattened_kernel = kernel.flatten()
    result = np.dot(windows_matrix, flattened_kernel)
    result_height = image.shape[0]-kernel.shape[0]+1
    result_width = image.shape[1]-kernel.shape[1]+1
    result = result.reshape(result_height, result_width)
    return result


def upsampling(img,g_size):
    h,w=img.shape
    alter_zeros=np.zeros((h*2,w*2))
    for i in range(h):
        for j in range(w):
            alter_zeros[i*2][j*2]=img[i][j]
    sigma_diff = math.sqrt(max((1.6 ** 2) - ((2 * 0.5) ** 2), 0.01))
    up2=convolve(alter_zeros,gaussian_matrix(g_size,sigma_diff))
    return up2
    

def gen_gauss(s,sigma):
    all_gauss=[]
    all_gauss.append(sigma)
    k=2**(1./s)
    
    for i in range(1,s+3):
        pre=(k**(i-1))*sigma
        all_gauss.append(math.sqrt((k*pre)**2-(pre)**2))
    all_gauss=np.array(all_gauss)
    return all_gauss

def gauss_img_in_octave(g_siz,img,filters):
    gauss_imgs=[]
    gauss_imgs.append(img)
    for i in filters[1:]:
        gauss_imgs.append(convolve(gauss_imgs[-1],gaussian_matrix(11,i)))
    gauss_imgs=np.array(gauss_imgs)
    return gauss_imgs

def diff_of_gauss(imgs):
    dog=[]
    for i in range(1,imgs.shape[0]):
        dog.append(imgs[i]-imgs[i-1])
    dog=np.array(dog)
    return dog


def check(arr1,arr2,arr3):
    
    if(np.all(arr1>=arr2[1,1]) and np.all(arr3>=arr2[1,1]) and arr2[0,0]>=arr2[1,1] and arr2[0,1]>=arr2[1,1] and arr2[0,2]>=arr2[1,1] and arr2[1,0]>=arr2[1,1] and arr2[1,2]>=arr2[1,1] and arr2[2,0]>=arr2[1,1] and arr2[2,1]>=arr2[1,1] and arr2[2,2]>=arr2[1,1]):
        return True
    elif(np.all(arr1<=arr2[1,1]) and np.all(arr3<=arr2[1,1]) and arr2[0,0]<=arr2[1,1] and arr2[0,1]<=arr2[1,1] and arr2[0,2]<=arr2[1,1] and arr2[1,0]<=arr2[1,1] and arr2[1,2]<=arr2[1,1] and arr2[2,0]<=arr2[1,1] and arr2[2,1]<=arr2[1,1] and arr2[2,2]<=arr2[1,1]):
        return True
    else:
        return False
    

def edge_points_removal(location):
    dxx=location[1,2]+location[1,0]-2*location[1,1]
    dyy=location[0,1]+location[2,1]-2*location[1,1]
    dxy=0.25*(location[0,2]+location[2,0]-location[0,0]-location[2,2])
    trace=dxx+dyy
    det=(dxx*dyy)-(dxy)**2
    r=10
    if((r*(trace**2))<(det*((r+1)**2))):
        return True
    return False



def edge_points_removal_wrong(location):
    dxx=location[1,2]+location[1,0]-2*location[1,1]
    dyy=location[0,1]+location[2,1]-2*location[1,1]
    dx=location[1,2]-location[1,0]
    dy=location[2,1]-location[0,1]
    dxy=dx*dy
    trace=dxx+dyy
    det=(dxx*dyy)-(dxy)**2
    r=10
    if((r*(trace**2))<(det*((r+1)**2))):
        return True
    return False

def points(dog,octave_number,s,unique_key_points):
    p=[]
    n=dog.shape[0]
    leave=s//2
    h,w=dog.shape[1:]
    keys_in_octave={}
    for i in range(0,n-2):
        # c=0
        for j in range(leave,h-leave):
            for k in range(leave,w-leave):
                if(dog[i+1,j,k]>0.0025 and edge_points_removal(dog[i+1,j-leave:j+leave+1,k-leave:k+leave+1]) and check(dog[i,j-leave:j+leave+1,k-leave:k+leave+1],dog[i+1,j-leave:j+leave+1,k-leave:k+leave+1],dog[i+2,j-leave:j+leave+1,k-leave:k+leave+1])):
                    unmapped_point=[j,k]
                    # c+=1
                    mapped_point=tuple([math.floor(j*(2**(octave_number-1))),math.floor(k*(2**(octave_number-1)))])
                    if i+1 not in keys_in_octave:
                        keys_in_octave[i+1]=[] 
                    keys_in_octave[i+1].append(unmapped_point)
                    unique_key_points.add(mapped_point)
    return keys_in_octave


def downsampling(img):
    h,w=img.shape
    down=np.zeros((int(h//2),int(w//2)))
    for i in range(int(h//2)):
        for j in range(int(w//2)):
            down[i][j]=img[i*2][j*2]
    return down


def remove_dupli(all_key_points):
    s=set()
    for i in all_key_points:
        for j in i:
            s.add(tuple(j))
    s=np.array(list(s))
    return s


def all_max_angels(orient_bins):
    principle_angles = []
    orient_bins /= np.max(orient_bins)
    original_orient_bins = orient_bins.copy()
    for i in np.where(orient_bins >= 0.8)[0]:
        t1, t2, t3 = (10 * (i - 1)) + 5, (10 * i) + 5, (10 * (i + 1)) + 5
        x = [t1, t2, t3]

        if 0 < i < 35:
            y = [original_orient_bins[i-1], original_orient_bins[i], original_orient_bins[i+1]]
        elif i == 0:
            y = [original_orient_bins[35], original_orient_bins[0], original_orient_bins[1]]
        elif i == 35:
            y = [original_orient_bins[34], original_orient_bins[35], original_orient_bins[0]]

        if y[1] > y[0] and y[1] > y[2]:
            coefficients = np.linalg.solve(np.vstack([np.array(x) ** 2, x, np.ones(len(x))]).T, y)
            # print()
            final_t = -(coefficients[1]) / (2 * coefficients[0])

            if i == 0 and final_t < 0:
                final_t += 360
            elif i == 35 and final_t > 360:
                final_t -= 360

            principle_angles.append(final_t)

    return principle_angles


def discriptor(p,gaussian_imgs,octave_no,sigma_diff):
    sigmas=sigma_diff.copy()
    sigmas=sigmas**2
    all_bins={}
    p_c=np.array([7,7])
    big_gauss=gaus_of_discriptor(16,8,p_c)
    # print("big gauss : ",big_gauss[:3,:3])
    for i,point in p.items():
        for p in point:
            h,w=gaussian_imgs[0].shape
            y=p[0]
            x=p[1]
            if(y-8<0 or y+10>h or x-8<0 or x+10>w):
                continue
            # sigma=np.sum(sigma_diff[0:i+1])*(2**octave_no)*1.5
            sigma=np.sqrt(np.sum(sigmas[0:i+1]))*1.5
            # print("sigma : ",sigma)
            gaus=gaus_of_discriptor(16,sigma,p_c)
            # print("small gauss : ",  gaus[:3,:3])
            one_point_bin=np.zeros(36)
            
            x_original=math.floor(x*(2**(octave_no-1)))
            y_original=math.floor(y*(2**(octave_no-1)))
            mapped_points=(y_original,x_original)
            x_conv=np.array([[-1,0,1]])
            y_conv=x_conv.T
            # print(i)
            dx=convolve_withoutpadding(gaussian_imgs[i,y-7:y+9,x-8:x+10],x_conv)
            dy=convolve_withoutpadding(gaussian_imgs[i,y-8:y+10,x-7:x+9],y_conv)
            angel=np.degrees(np.arctan(np.divide(dy,dx)))
            angel[angel<0]+=360
            magnitude=np.sqrt((dx**2)+(dy**2))
            # print("dx : ",dx)
            # print("dy : ",dy)
            gauss_mag=magnitude*gaus
            for m in range(16):
                for n in range(16):
                    one_point_bin[int(angel[m,n]//10)]+=gauss_mag[m,n]
            principal_angels=all_max_angels(one_point_bin)
            for angels in principal_angels:
                disk=np.array([])
                for m in range(0,16,4):
                    for n in range(0,16,4):
                        final_bin=np.zeros(8)
                        sub_magnitude=magnitude[m:m+4,n:n+4].copy()
                        sub_angel=angel[m:m+4,n:n+4].copy()
                        sub_gauss=big_gauss[m:m+4,n:n+4].copy()
                        sub_magnitude*=sub_gauss
                        sub_angel-=angels
                        sub_angel[sub_angel<0]+=360
                        for p in range(4):
                            for q in range(4):
                                # if(sub_angel[p,q]>360):
                                    # print(max_pos)
                                    # print(mapped_points)
                                    # print(angels)
                                    # print(sub_angel)
                                    # print(angel)
                                final_bin[int(sub_angel[p,q]//45)]+=sub_magnitude[p,q]
                        disk=np.concatenate((disk,final_bin))
                if mapped_points not in all_bins :
                    all_bins[mapped_points]=[]
                all_bins[mapped_points].append(disk)
    return all_bins



def merge_values(val1, val2):
    if not isinstance(val1, list):
        val1 = [val1]
    if not isinstance(val2, list):
        val2 = [val2]
    return val1 + val2

def merge(dict1, dict2):
    combined_dict = {}
    for key in set(dict1) | set(dict2):  # Union of keys from both dictionaries
        val1 = dict1.get(key, [])
        val2 = dict2.get(key, [])
        combined_dict[key] = merge_values(val1, val2)
    return combined_dict


def keypoints(img_path,octave_start=1,octave_end=10,s=3):
    img=cv2.imread(img_path)
    gray = (cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))/255
    levelup=upsampling(gray,5)
    temp=levelup.copy()
    g_size=5
    sigma=1.6
    maxkernel=3
    all_key_points=[]
    g_filters=[]
    g_imgs=[]
    dogs=[]
    allfilters=gen_gauss(s,sigma)
    global_discriptors={}
    unique_key_points=set()
    for i in range(5):
        gauss_img=gauss_img_in_octave(g_size,temp,allfilters)
        # print('g imges genrated')
        temp=downsampling(gauss_img[-3])
        # print('downsampled')
        dog=diff_of_gauss(gauss_img)
        # print('dog generated')
        keys_in_octave=points(dog,i,maxkernel,unique_key_points)
        local_discriptors=discriptor(keys_in_octave,gauss_img,i,allfilters)
        # print('key points found')
        # g_filters.append(allfilters)
        # g_imgs.append(gauss_img)
        # dogs.append(dog)
        global_discriptors=merge(global_discriptors,local_discriptors)
    # keys=remove_dupli(all_key_points)
        
    return(unique_key_points,global_discriptors)


def get_points_and_discriptors(final_discriptors):
    point=[]
    dis=[]
    for i,d in final_discriptors.items():
        for j in range(len(d)):
            point.append(i)
            dis.append(d[j])
    return point,dis


def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2)**2))

def match_keypoints(descriptors1, descriptors2,point_list1,point_list2,shape1,shape2, threshold=0.6):
    matches = []
    h1,w1=shape1
    h2,w2=shape2
    for i, descriptor1 in enumerate(descriptors1):
        if(point_list1[i][1]>w1//2):
            best_match_index = None
            best_distance = float('inf')
            second_best_distance = float('inf')

            for j, descriptor2 in enumerate(descriptors2):
                if(point_list2[j][1]<w2//2):
                    distance = euclidean_distance(descriptor1, descriptor2)
                    if distance < best_distance:
                        second_best_distance = best_distance
                        best_distance = distance
                        best_match_index = j
                    elif distance < second_best_distance:
                        second_best_distance = distance

            if best_distance <= threshold * second_best_distance:
                matches.append((i, best_match_index))
            # print((i, best_match_index))
    return matches

def random_color():
    return (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))


def find_diff(t1,t2,confidence):
    diff=t1-t2
    min_euc=5
    d=np.sqrt(diff[:,0]**2+diff[:,1]**2)
    
    total=d.shape[0]
    inliers=np.count_nonzero(d < min_euc)
    percent_of_inlier=inliers/total
    if(percent_of_inlier>=confidence):
        # print(percent_of_inlier)
        inlier_idx = np.where(d<min_euc)[0]
        return inlier_idx,percent_of_inlier
    else:
        # print(percent_of_inlier)
        return [],0
    


def gen_H(pts1,pts2):
    # pts1=np.array(pts1)
    # pts2=np.array(pts2)
    # print(pts1.shape,pts2.shape)
    A=np.array([])
    for j in range(len(pts1)):
        row1=np.array([pts1[j][0],pts1[j][1],1,0,0,0,-pts2[j][0]*pts1[j][0],-pts2[j][0]*pts1[j][1],-pts2[j][0]])
        row2=np.array([0,0,0,pts1[j][0],pts1[j][1],1,-pts2[j][1]*pts1[j][0],-pts2[j][1]*pts1[j][1],-pts2[j][1]])
        if A.size == 0:
            A = np.vstack((row1, row2))
        else:
            A = np.vstack((A, row1))
            A = np.vstack((A, row2))
    A_trans=np.dot(A.T,A)
    eigen_val, eigen_vect = np.linalg.eig(A_trans)
    smallest_idx = np.argmin(eigen_val)
    H_flattend = eigen_vect[:, smallest_idx]
    H=H_flattend.reshape((3,3))
    return H



def homo(keypoints1,keypoints2):
    s=len(keypoints1)
    k=100000
    global_source=[]
    global_dest = []
    confidence=0
    
    for i in range(k):
        mask=np.zeros(len(keypoints1))
        new_pts = random.sample(range(s), 4)
        # print("4 points location :",new_pts)
        pts1=[keypoints1[j] for j in new_pts ]
        pts2=[keypoints2[j] for j in new_pts ]
        other_than_four=[j for j in range(s) if j not in new_pts]
        s1=set()
        for x in range(4):
            temp=(tuple(pts1[x]),tuple(pts2[x]))
            s1.add(temp)
        if(len(s1)<4):
            continue

        other_pts_in1=[keypoints1[j] for j in range(s) if j not in new_pts]
        other_pts_in2=[keypoints2[j] for j in range(s) if j not in new_pts]
        array_of1 = np.array(other_pts_in1)
        array_of2 = np.array(other_pts_in2)
        ones_column = np.ones((array_of1.shape[0], 1))
        result_matrix = np.hstack((array_of1, ones_column))
        # print(result_matrix)
        # print(pts1,pts2)
        H=gen_H(pts1,pts2)
        # print(H)
        transformed_pts=np.dot(result_matrix,H.T)
        # print(transformed_pts)
        transformed_pts[:,0]=transformed_pts[:,0]/transformed_pts[:,2]
        transformed_pts[:,1]=transformed_pts[:,1]/transformed_pts[:,2]
        set_of_inliers,conf=find_diff(transformed_pts[:,:2],array_of2,confidence)
        if(conf>confidence):
            confidence=conf
            # print(mask.shape)
            # print(len(set_of_inliers))
            # print(set_of_inliers)
            # print(len(other_than_four))
            # print(other_than_four)
            global_source=[other_pts_in1[idx] for idx in set_of_inliers]
            for item in pts1:
                global_source.append(item)
            global_dest=[other_pts_in2[idx] for idx in set_of_inliers]
            for item in pts2:
                global_dest.append(item)
            for item in set_of_inliers:
                mask[other_than_four[item]]=1
            for item in new_pts:
                mask[item]=1
            


    print("confidence ",confidence)
    # print("keypoints : ",keypoints1)
    # print("global source : " ,global_source)
    # print("global dest : " ,global_dest) 
    global_source = [list(arr) for arr in global_source]
    global_dest = [list(arr) for arr in global_dest]
    # global_source=(np.array(global_source)).reshape(-1,2)
    # global_dest=(np.array(global_dest)).reshape(-1,2)
    # print(global_source)
    # print(global_dest) 
    H=gen_H(global_source,global_dest)
    return H,mask
        



def warpPerspective(original_image, adjusted_homography_matrix, calc_warp_w1, calc_warp_h1):
    H_inv = np.linalg.inv(adjusted_homography_matrix)
    
    warped_image = np.ones((calc_warp_h1, calc_warp_w1, 3), dtype=np.uint8) * 255
    
    for y in range(calc_warp_h1):
        for x in range(calc_warp_w1):
            point = np.dot(H_inv, [x, y, 1]) / np.dot(H_inv, [x, y, 1])[2]
            px, py = int(round(point[0])), int(round(point[1]))
            if 0 <= px < original_image.shape[1] and 0 <= py < original_image.shape[0]:
                warped_image[y, x] = original_image[py, px]
            
    return warped_image




def stitch_two_imgs(new_h,new_w,min_y,warp_h2,warp_w2,trans_x,trans_y,warp_image2,img1,h1,w1,flag):
    extended_image = np.ones((new_h, new_w, 3), dtype=np.uint8) * 255
    if flag==0:
        if min_y<=0 and flag==0:
        
            extended_image[:warp_image2.shape[0], :warp_image2.shape[1]] = warp_image2
            print("h,w=",extended_image.shape)
            extended_image[0+trans_y:0+trans_y+h1, trans_x:trans_x+w1] = img1

        elif flag==0:
        
            extended_image[-trans_y:-trans_y+warp_image2.shape[0], :warp_image2.shape[1]] = warp_image2
            print("h,w=",extended_image.shape)
            extended_image[:h1, trans_x:trans_x+w1] = img1
            trans_y = 0
    else:

        if min_y<=0 and flag!=0:
        
            extended_image[:warp_image2.shape[0], -trans_x:-trans_x+warp_image2.shape[1]] = warp_image2
            print("h,w=",extended_image.shape)
            extended_image[trans_y:trans_y+h1, :w1] = img1
        elif flag!=0:
        
            extended_image[-trans_y:-trans_y+warp_image2.shape[0], -trans_x:-trans_x+warp_image2.shape[1]] = warp_image2
            print("h,w=",extended_image.shape)
            extended_image[:h1, :w1] = img1
            trans_y = 0
    return extended_image,trans_y



def left_stitch(start,end,x_translation,y_translation):
    left_img=images[start].copy()
    for i in range(start+1,end):
        img2=images[i].copy()
        h2,w2=img2.shape[:2]
        h1,w1=left_img.shape[:2]
        # plt.imshow(left_img)
        # plt.show()
        # plt.imshow(img2)
        # plt.show()
        matched_points_img1=[]
        matched_points_img2=[]
        for j in all_matches[i]:
            y,x=keypoints_list[i][j[0]]
            matched_points_img1.append((x,y))
            y,x=keypoints_list[i+1][j[1]]
            matched_points_img2.append((x,y))
        matched_points_img1=np.array(matched_points_img1)
        matched_points_img1[:,0]+=x_translation
        
        matched_points_img2=np.array(matched_points_img2)
        
        matched_points_img1[:,1]+=y_translation


        homography, mask = homo(matched_points_img1, matched_points_img2)
#         homography, mask = cv2.findHomography(matched_points_img1, matched_points_img2, cv2.RANSAC)
        print(homography)
        transformed_corners = np.dot(homography, (np.array([[0, 0, 1], [0, h1 - 1, 1], [w1 - 1, h1 - 1, 1], [w1 - 1, 0, 1]])).T).T
        normalized_corners = [[corner[0] / corner[2], corner[1] / corner[2]] for corner in transformed_corners]

        min_x = min(corner[0] for corner in normalized_corners)
        trans_x = -math.floor(min_x)
        min_y = min(corner[1] for corner in normalized_corners)
        max_x = max(corner[0] for corner in normalized_corners)
        max_y = max(corner[1] for corner in normalized_corners)

        # calc_warp_w1 = math.ceil(max_x) - math.floor(min_x)
        # calc_warp_h1 = math.ceil(max_y) - math.floor(min_y)
        
        trans_y = -math.floor(min_y)
        adjusted_homography_matrix = np.dot(np.array([[1, 0, -math.floor(min_x)], [0, 1, -math.floor(min_y)], [0, 0, 1]]), homography)
        print(adjusted_homography_matrix)
        # plot_matched_points(adjusted_homography_matrix,matched_points_img1,matched_points_img2,left_img,img2,mask)
        warp_image1 = warpPerspective(left_img, adjusted_homography_matrix, math.ceil(max_x) - math.floor(min_x), math.ceil(max_y) - math.floor(min_y))
        # warp_image1 = cv2.warpPerspective(left_img, adjusted_homography_matrix, (calc_warp_w1, calc_warp_h1), borderValue=(255, 255, 255))
        warp_h1, warp_w1 = warp_image1.shape[:2]
        # plt.imshow(warp_image1)
        # plt.title("warped")
        # plt.show()
        # warp_keypoints1 = np.dot(adjusted_homography_matrix, (np.hstack((keypoints1, np.ones((keypoints1.shape[0], 1))))).T).T
        # warp_keypoints1 = [[x / z, y / z] for x, y, z in warp_keypoints1]
        new_w = w2 + trans_x
        new_h = max(math.ceil(max_y),h2)-min(math.floor(min_y),0)
        extended_image,trans_y=stitch_two_imgs(new_h,new_w,min_y,warp_h1,warp_w1,trans_x,trans_y,warp_image1,img2,h2,w2,0)
        x_translation=trans_x
        y_translation=trans_y
        left_img=extended_image.copy()
        # plt.imshow(left_img)
        # plt.show()
    return left_img,x_translation,y_translation


def right_stitch(start,end,x_translation,y_translation):
    right_img=images[end-1].copy()
    for i in range(end-2,start-1,-1):
        print(i)
        img1=images[i].copy()
        # plt.imshow(right_img)
        # plt.show()
        # plt.imshow(img1)
        # plt.show()
        matched_points_img1=[]
        matched_points_img2=[]
        for j in all_matches[i]:
            y,x=keypoints_list[i][j[0]]
            matched_points_img1.append((x,y))
            y,x=keypoints_list[i+1][j[1]]
            matched_points_img2.append((x,y))
        matched_points_img1=np.array(matched_points_img1)
        matched_points_img2=np.array(matched_points_img2)
        matched_points_img2[:,1]+=y_translation

        h2,w2=right_img.shape[:2]
        h1,w1=img1.shape[:2]

        homography, mask = homo(matched_points_img2,matched_points_img1)
#         homography, mask = cv2.findHomography( matched_points_img2,matched_points_img1, cv2.RANSAC)
        print(homography)
        transformed_corners = np.dot(homography, (np.array([[0, 0, 1], [0, h2 - 1, 1], [w2 - 1, h2 - 1, 1], [w2 - 1, 0, 1]])).T).T
        normalized_corners = [[corner[0] / corner[2], corner[1] / corner[2]] for corner in transformed_corners]

        min_x = min(corner[0] for corner in normalized_corners)
        trans_x = -math.floor(min_x)
        min_y = min(corner[1] for corner in normalized_corners)
        max_x = max(corner[0] for corner in normalized_corners)
        max_y = max(corner[1] for corner in normalized_corners)

        # calc_warp_w1 = math.ceil(max_x) - math.floor(min_x)
        # calc_warp_h1 = math.ceil(max_y) - math.floor(min_y)
        
        trans_y = -math.floor(min_y)
        adjusted_homography_matrix = np.dot(np.array([[1, 0, -math.floor(min_x)], [0, 1, -math.floor(min_y)], [0, 0, 1]]), homography)
        print(adjusted_homography_matrix)
        # plot_matched_points(adjusted_homography_matrix,matched_points_img1,matched_points_img2,left_img,img2,mask)
        # warp_image2 = cv2.warpPerspective(right_img, adjusted_homography_matrix, (calc_warp_w1, calc_warp_h1), borderValue=(255, 255, 255))
        warp_image2 = warpPerspective(right_img, adjusted_homography_matrix, math.ceil(max_x) - math.floor(min_x), math.ceil(max_y) - math.floor(min_y))
        warp_h2, warp_w2 = warp_image2.shape[:2]
        # plt.imshow(warp_image2)
        # plt.title("warped")
        # plt.show()
        # warp_keypoints1 = np.dot(adjusted_homography_matrix, (np.hstack((keypoints1, np.ones((keypoints1.shape[0], 1))))).T).T
        # warp_keypoints1 = [[x / z, y / z] for x, y, z in warp_keypoints1]
        new_w = warp_w2 - trans_x
        new_h = max(math.ceil(max_y),h1)-min(math.floor(min_y),0)
        extended_image,trans_y=stitch_two_imgs(new_h,new_w,min_y,warp_h2,warp_w2,trans_x,trans_y,warp_image2,img1,h1,w1,1)
        # x_translation=trans_x
        y_translation=trans_y
        right_img=extended_image.copy()
        # plt.imshow(right_img)
        # plt.show()
    return right_img,y_translation

def stitch_right_left(p1,p2,left_img,right_img1,l_t_x,l_t_y,r_t_y):
    matched_points_img1=[]
    print(p1,p2)
    right_img=right_img1.copy()
    img1=left_img.copy()
    # plt.imshow(right_img)
    # plt.show()
    # plt.imshow(img1)
    # plt.show()
    matched_points_img2=[]
    for j in all_matches[p1]:
        y,x=keypoints_list[p1][j[0]]
        matched_points_img1.append((x,y))
        y,x=keypoints_list[p2][j[1]]
        matched_points_img2.append((x,y))
    matched_points_img1=np.array(matched_points_img1)
    matched_points_img1[:,0]+=l_t_x
    h2,w2=right_img.shape[:2]
    matched_points_img1[:,1]+=l_t_y
    matched_points_img2=np.array(matched_points_img2)
    h1,w1=img1.shape[:2]
    matched_points_img2[:,1]+=r_t_y
    homography, mask = homo(matched_points_img2,matched_points_img1)
#     homography, mask = cv2.findHomography( matched_points_img2,matched_points_img1, cv2.RANSAC)
    print(homography)
    transformed_corners = np.dot(homography, (np.array([[0, 0, 1], [0, h2 - 1, 1], [w2 - 1, h2 - 1, 1], [w2 - 1, 0, 1]])).T).T
    normalized_corners = [[corner[0] / corner[2], corner[1] / corner[2]] for corner in transformed_corners]
    min_x = min(corner[0] for corner in normalized_corners)
    trans_x = -math.floor(min_x)
    max_x = max(corner[0] for corner in normalized_corners)
    calc_warp_w1 = math.ceil(max_x) - math.floor(min_x)

    min_y = min(corner[1] for corner in normalized_corners)
    max_y = max(corner[1] for corner in normalized_corners)
    
    calc_warp_h1 = math.ceil(max_y) - math.floor(min_y)
    
    trans_y = -math.floor(min_y)
    adjusted_homography_matrix = np.dot(np.array([[1, 0, trans_x], [0, 1, trans_y], [0, 0, 1]]), homography)
    print(adjusted_homography_matrix)
    # plot_matched_points(adjusted_homography_matrix,matched_points_img1,matched_points_img2,left_img,img2,mask)
    # warp_image2 = cv2.warpPerspective(right_img, adjusted_homography_matrix, (calc_warp_w1, calc_warp_h1), borderValue=(255, 255, 255))
    warp_image2 = warpPerspective(right_img, adjusted_homography_matrix, calc_warp_w1, calc_warp_h1)
    warp_h2, warp_w2 = warp_image2.shape[:2]
    # plt.imshow(warp_image2)
    # plt.title("warped")
    # plt.show()
    # warp_keypoints1 = np.dot(adjusted_homography_matrix, (np.hstack((keypoints1, np.ones((keypoints1.shape[0], 1))))).T).T
    # warp_keypoints1 = [[x / z, y / z] for x, y, z in warp_keypoints1]
    new_w = warp_w2 - trans_x
    new_h = max(math.ceil(max_y),h1)-min(math.floor(min_y),0)
    extended_image,trans_y=stitch_two_imgs(new_h,new_w,min_y,warp_h2,warp_w2,trans_x,trans_y,warp_image2,img1,h1,w1,1)
    # x_translation=trans_x
    y_translation=trans_y
    right_img=extended_image.copy()
    # plt.imshow(right_img)
    # plt.show()
    return right_img




def stitch(num_imgs):
    if(num_imgs%2==0):

        left_img,l_t_x,l_t_y=left_stitch(max((((num_imgs)//2)-1)-2,0),num_imgs//2,0,0)
        right_img,r_t_y=right_stitch((num_imgs+1)//2,min(((num_imgs)//2)+1,num_imgs-1)+1,0,0)
        complete=stitch_right_left(num_imgs//2-1,num_imgs//2,left_img,right_img,l_t_x,l_t_y,r_t_y)
    else:
        left_img,l_t_x,l_t_y=left_stitch(max(((num_imgs)//2)-2,0),(num_imgs//2)+1,0,0)
        right_img,r_t_y=right_stitch((num_imgs)//2+1,min((((num_imgs)//2)+1)+1,num_imgs-1)+1,0,0)
        complete=stitch_right_left(num_imgs//2,num_imgs//2+1,left_img,right_img,l_t_x,l_t_y,r_t_y)
    return complete



def capture_frames_at_times(video_path, output_folder, num_frames=5):
    cap = cv2.VideoCapture(video_path)
    os.makedirs(output_folder, exist_ok=True)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    intervals = np.linspace(0, frame_count - 1, num_frames, dtype=int)
    
    for i, interval in enumerate(intervals):
        cap.set(cv2.CAP_PROP_POS_FRAMES, interval)
        ret, frame = cap.read()
        if ret:
            output_path = os.path.join(output_folder, f"{i + 1}.jpg")
            cv2.imwrite(output_path, frame)
    
    cap.release()




def paranoma(folder_path):
    all_imgs=os.listdir(folder_path)
    all_imgs=np.sort(all_imgs)
    dis=[]
    for i in all_imgs:
        img_path=folder_path+'/'+i
        all_key_points,final_discriptors=keypoints(img_path)
        dis.append(final_discriptors)
        orig_img=cv2.imread(img_path)
        rgb_image = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        images.append(rgb_image.copy())
        # for keypoint in all_key_points:
        #     cv2.circle(rgb_image, (int(keypoint[1]),int(keypoint[0])), 5, (255,0,0), -1)
        # plt.imshow(rgb_image)
        # # plt.axis('off')
        # plt.show()  
    for i in range(len(dis)):
        p,d=get_points_and_discriptors(dis[i])
        keypoints_list.append(p)
        descriptors_list.append(d)
        # print(len(p),len(d))
    num_images=len(all_imgs)
    
    for i in range(num_images-1):
        matches = match_keypoints(descriptors_list[i], descriptors_list[i+1],keypoints_list[i],keypoints_list[i+1],images[i].shape[:2],images[i+1].shape[:2])
        all_matches.append(matches)
    c=stitch(len(images))
    return c

images = []
keypoints_list = []
descriptors_list = []
all_matches = []

if(sys.argv[1]=="1"):
    folder_path=sys.argv[2]
    panaroma_img=paranoma(folder_path)
    saveto=sys.argv[3]+"/panoroma_image.jpg"
    cv2.imwrite(saveto, cv2.cvtColor(panaroma_img, cv2.COLOR_RGB2BGR))
elif(sys.argv[1]=="2"):
    video_path=sys.argv[2]
    folder_path=os.path.join(sys.argv[3], 'video_to_images')
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    capture_frames_at_times(video_path,folder_path)
    panaroma_img=paranoma(folder_path)
    saveto=sys.argv[3]+"/panoroma_image.jpg"
    cv2.imwrite(saveto, cv2.cvtColor(panaroma_img, cv2.COLOR_RGB2BGR))
