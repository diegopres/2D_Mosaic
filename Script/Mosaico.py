# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 09:20:28 2019

@author: uidv7259
"""

from skimage import io
import os
import tkinter as tk
import matplotlib.pyplot as plt
from skimage.transform import resize
import pandas as pd
import ast
import Kmeans as km
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import random
import time

class Mosaico:
    def __init__(self, main_path, direct):
        self.image = io.imread(main_path)
        self.height, self.width, self.ch = self.image.shape
        self.main_area =  self.height * self.width
        self.image_list = []
        self.image_cluster = None
        self.cluster_map = pd.DataFrame()
        directory = os.fsencode(direct)
        for file in os.listdir(directory):
             filename = os.fsdecode(file)
             if filename.endswith(".jpg") or filename.endswith(".JPG"):
                 self.image_list.append((filename, io.imread(direct + filename)))
        self.nxy = None
        self.visited_img = [False] * len(self.image_list)
        self.complete_center = None
        self.counter = 0
        
    def show_main_image(self, second = False):
        if second:
            fig = plt.figure(frameon=False)
            fig.set_size_inches(6,7)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(self.image)
        else:
            plt.imshow(self.image)
        
    
    def show_image(self, n):
        plt.imshow(self.image_list[n])
    
    def resize_image(self, img, nx = None, ny = None):
        if nx == None and ny == None:
            nx = self.nxy
            ny = self.nxy
        new_image = resize(img, (nx, ny))
        rescaled_image = new_image * 255
        new_image = rescaled_image.astype(np.uint8)
        return new_image
    
    def get_dominant_color_rgb_mean(self, img):
        new_shape_img = img.reshape((-1,3))
        size =  len(new_shape_img)
        sum_array = np.sum(new_shape_img, axis =0)
        return [sum_array[0]//size,  sum_array[1]//size,  sum_array[2]//size]
        
    
    def classify_images(self, img_array, nclust, it, plot_3d = False, listnames = None):
        self.image_cluster = km.K_means(n_clusters = nclust, iterations = it)
        self.image_cluster.fit(img_array)
        self.cluster_map['values'] = img_array
        self.cluster_map['labels'] = self.image_cluster.labels_
        self.cluster_map['filename'] = listnames
        self.complete_center = [False] * len(self.image_cluster.cluster_centers_)

        #To plot dominant colors and centers
        if plot_3d:
            colors = ['red', 'green', 'blue', 'cyan', 'orange']
            color_list = []
            for label in self.image_cluster.labels_:
                color_list.append(colors[int(label)])
            fig = plt.figure(2)
            ax = Axes3D(fig)
            img_array =  np.array(img_array)
            centers = self.image_cluster.cluster_centers_
            ax.scatter(img_array[:, 0], img_array[:, 1], img_array[:, 2], c = color_list)
            ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], marker='*', c='#050505', s=500)

    
    def find_best_match(self, coor):
        min_ = 442 #Maximum possible distance
        min_label = None
        coor_x, coor_y = coor
        y0 = coor_y * self.nxy
        y1 = y0 + self.nxy
        x0 = coor_x * self.nxy
        x1 = x0 + self.nxy
        block = np.array(self.image[y0:y1, x0:x1])
        
        dominant_color = self.get_dominant_color_rgb_mean(block)
        rd, gd, bd = dominant_color
        
        for i, rgb in enumerate(self.image_cluster.cluster_centers_):
            r, g, b = rgb
            d = ((r-rd)**2 + (g-gd)**2 + (b-bd)**2)**0.5
            if d < min_:
                min_ = d
                min_label = i
        
        min_ = 442
        min_index = None
        temp_dataFrame = self.cluster_map[self.cluster_map.labels == min_label]
        for i, row in temp_dataFrame.iterrows():
            r, g, b = row['values']
            d = ((r-rd)**2 + (g-gd)**2 + (b-bd)**2)**0.5
            if d < min_:
                min_ = d
                min_index = i
    
        return min_index

    def find_best_match_2(self, coor):
        coor_x, coor_y = coor
        y0 = coor_y * self.nxy
        y1 = y0 + self.nxy
        x0 = coor_x * self.nxy
        x1 = x0 + self.nxy
        block = np.array(self.image[y0:y1, x0:x1])
        min_index = None
        

        
        
        dominant_color = self.get_dominant_color_rgb_mean(block)
        rd, gd, bd = dominant_color
        
        while(min_index == None):
#            print(self.complete_center)
#            print(self.visited_img)
            min_ = 442 #Maximum possible distance
            min_label = None
            if self.counter == (len(self.image_cluster.cluster_centers_)):
                self.counter = 0
                for i in range(len(self.image_cluster.cluster_centers_)):
                    self.complete_center[i] = False 
                for i in range(len(self.image_list)):
                    self.visited_img[i] = False
            for i, rgb in enumerate(self.image_cluster.cluster_centers_):
                r, g, b = rgb
                d = ((r-rd)**2 + (g-gd)**2 + (b-bd)**2)**0.5
                if d < min_ and not self.complete_center[i]:
                    min_ = d
                    min_label = i
                    

            min_ = 442
            temp_dataFrame = self.cluster_map[self.cluster_map.labels == min_label]
            for i, row in temp_dataFrame.iterrows():
                r, g, b = row['values']
                d = ((r-rd)**2 + (g-gd)**2 + (b-bd)**2)**0.5
                if d < min_ and not self.visited_img[i]:
                    min_ = d
                    min_index = i
            

            if min_index == None:
                self.complete_center[min_label] = True
                self.counter += 1
                print(self.counter)
            else:
                self.visited_img[min_index] = True
                break
            
                
            
        return min_index
    
    def replace_img_section(self, img, coor):
        coor_x, coor_y = coor
        y0 = coor_y * self.nxy
        y1 = y0 + self.nxy
        x0 = coor_x * self.nxy
        x1 = x0 + self.nxy
        if y1 > self.height and x1 > self.width:
            y = self.height - y0
            x = self.width - x0
            self.image[y0:self.height, x0:self.width] = img[0:y, 0:x]
        elif y1 > self.height:
            y = self.height - y0
            self.image[y0:self.height, x0:x1] = img[0:y, :]
        elif x1 > self.width:
            x = self.width - x0
            self.image[y0:y1, x0:self.width] = img[:, 0:x]
        else:
            self.image[y0:y1, x0:x1] = img
        
    def get_RGB_pixel(self, pix):
        print(self.image)
        
        
        
#A-posteriori
def createIntArray(size):
    return [[random.randint(0,255), random.randint(0,255), random.randint(0,255)]  for x in range(size)]

def datosParaGraficarKmeans(minSize=2,maxSize=100, step=1, runs=200, nclust = 5, it = 10):
    totalC=[] #arreglo con el promedio de comparaciones para cada tamaño del arreglo
    totalM=[] #arreglo con el promedio de movimientos
    img_clstr = km.K_means(n_clusters = nclust, iterations = it)
    
    for size in range(minSize, maxSize, step):
        sum_mov = 0
        sum_comp = 0
        for i in range(runs):
            test_array = createIntArray(size)
            img_clstr.fit(test_array)
            sum_mov += img_clstr.mov
            sum_comp += img_clstr.comp
        totalM.append(sum_mov/runs)
        totalC.append(sum_comp/runs)
    
    return totalC, totalM

def a_posteriori():
    minSize = 5
    maxSize = 200
    step = 5
    runs = 20
    
    C,M = datosParaGraficarKmeans(minSize,maxSize,step,runs)
    
    x=[x for x in range(minSize, maxSize, step)]
    
    coefficients_C = np.polyfit(x, C, 1)
    coefficients_M = np.polyfit(x, M, 1)
    print(coefficients_C, coefficients_M)
    
    p_C = np.poly1d(coefficients_C)
    p_M = np.poly1d(coefficients_M)
    xp = np.linspace(1, maxSize, maxSize)
    
    plt.plot(x, C,'g*', label="comparaciones c0x^2+c1x+c2")
    plt.plot(xp, p_C(xp),'r-', label="tendencia comp "+ str(round(coefficients_C[0],2))+" x^2 + " + str(round(coefficients_C[1],2))
             +" x + " + str(round(coefficients_C[1],2)))
    plt.plot(x, M,'bs', label="movimientos k0x^2+k1x+k2") 
    plt.plot(xp, p_M(xp),'c-', label="tendencia mov "+ str(round(coefficients_M[0],2))+" x^2 + " + str(round(coefficients_M[1],2))
             +" x + " + str(round(coefficients_M[1],2))) 
    plt.legend()
    plt.show()
        
def start_main(main_img, folder_imgs, size = 25, k = 5, it = 10, no_repeat = False):
#    main_img = "c:/Users/uidv7259/Documents/Maestria/1er_Semestre/Algoritmos/Proyecto/Images/eagle.jpg"
#    folder_imgs = "c:/Users/uidv7259/Documents/Maestria/1er_Semestre/Algoritmos/Proyecto/Images/Image_Folder_App/"
    if "\\" in main_img:
        main_img = main_img.replace("\\", "/")
    if "\\" in folder_imgs:
        folder_imgs = folder_imgs.replace("\\", "/")
    if not folder_imgs.endswith("/"):
        folder_imgs += "/"
        
    my_means_list = []
    img_list = []
    
    my_mosaic = Mosaico(main_img, folder_imgs)
    my_mosaic.nxy = size
    
    plt.figure(0)
    my_mosaic.show_main_image()
    
    f = open("images.txt",'r')
    list_names = []
    img_dict = {}
    line = f.readline()
    while line:
        f_name, dom_color = line.split("!")
        dom_color = ast.literal_eval(dom_color)
        img_dict[f_name] = np.array(dom_color)
        line = f.readline()
    f.close()
    for filename, img in my_mosaic.image_list:
        new_img =  np.array(my_mosaic.resize_image(img))
        dom_color = img_dict.get(filename, None)
        list_names.append(filename)
        if dom_color.any() != None:
            my_means = dom_color
        else:
            my_means = my_mosaic.get_dominant_color_rgb_mean(new_img)
#        f.write(filename + "," + str(my_means) + "\n")
        my_means_list.append(my_means)
        img_list.append((filename, new_img))
        
    
    start = time.time()
    my_mosaic.classify_images(my_means_list, k, it, False, list_names) #Get clusters of centroids
    print("Finish classificaton")
    end = time.time()
    print("Elapsed time img classification", (end - start))
    
    x_axis = my_mosaic.width//my_mosaic.nxy
    y_axis = my_mosaic.height//my_mosaic.nxy
    
    for x in range(x_axis + 1):
        for y in range(y_axis + 1):
            coor = (x,y)
            if no_repeat:
                best_img_index = my_mosaic.find_best_match_2(coor)
            else:
                best_img_index = my_mosaic.find_best_match(coor)
            if best_img_index != None:
                my_mosaic.replace_img_section(img_list[best_img_index][1], coor)
    

    print("Done!")
    
#    f2 = open("images_wClass.txt",'w+')
#    for i, row in my_mosaic.cluster_map.iterrows():
#        f2.write(filename + "," + str(my_means) + "\n")
#    f2.close()
    
    my_mosaic.show_main_image(second = True)
    


if __name__ == '__main__':
    window = tk.Tk()
    window.title("Mosaic App")
    window.geometry('600x200')
    
    lbl1 = tk.Label(window, text="Imagen Principal:")
    lbl1.grid(column=0, row=0)
    txt1 = tk.Entry(window,width=40)
    txt1.grid(column=10, row=0)
    
    lbl2 = tk.Label(window, text="Ruta imágenes:")
    lbl2.grid(column=0, row=1)
    txt2 = tk.Entry(window,width=40)
    txt2.grid(column=10, row=1)
    
    lbl3 = tk.Label(window, text="NxN")
    lbl3.grid(column=0, row=2)
    txt3 = tk.Entry(window,width=10)
    txt3.grid(column=10, row=2)
    
    #Kmeans
    lbl4 = tk.Label(window, text="K clusters:")
    lbl4.grid(column=0, row=4)
    txt4 = tk.Entry(window,width=10)
    txt4.grid(column=10, row=4)
    
    lbl5 = tk.Label(window, text="Iteraciones:")
    lbl5.grid(column=0, row=5)
    txt5 = tk.Entry(window,width=10)
    txt5.grid(column=10, row=5)
    
    def clicked():
        start_main(txt1.get(), txt2.get(), int(txt3.get()), int(txt4.get()), int(txt5.get()), True)
    def analysis():
        a_posteriori()
    
    btn = tk.Button(window, text="Crear Mosaico", command=clicked)
    btn.grid(column=25, row=0)
    
    btn2 = tk.Button(window, text="A-posteriori", command=analysis)
    btn2.grid(column=25, row=1)
    
    window.mainloop()