import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

def red_white_blue():    
    bottom = cm.get_cmap('YlOrRd', 128)
    #print(bottom)
    bottom_w = bottom(np.linspace(0, 1, 128)) #256
    w = np.array([1, 1, 1, 1])
    bottom_w[0:9, :] = w 
    top = cm.get_cmap('Blues_r', 128)
    top_w = top(np.linspace(0, 1, 128)) #256
    top_w[-1, :] = w 
    top_w[-1-10:-1,:]= w
    
    #newcolors = np.vstack((top_w(np.linspace(0, 1, 128)),\
    #                       bottom_w(np.linspace(0, 1, 128))))
    newcolors = np.vstack((top_w,bottom_w))
    rwb = ListedColormap(newcolors, name='RWB')
    #print(rwb)
    return rwb

def red_white_blue_21():    
    bottom = cm.get_cmap('YlOrRd', 128)
    #print(bottom)
    bottom_w = bottom(np.linspace(0, 1, 11)) #256
    w = np.array([1, 1, 1, 1])
    bottom_w[0, :] = w 
    top = cm.get_cmap('Blues_r', 128)
    top_w = top(np.linspace(0, 1, 11)) #256
    top_w[-1, :] = w 
    #top_w[-1-10:-1,:]= w
    
    newcolors = np.vstack((top_w,bottom_w))
    rwb = ListedColormap(newcolors, name='RWB_21')
    #print(newcolors)
    return rwb


def red_blue():    
    bottom = cm.get_cmap('YlOrRd', 128)
    top = cm.get_cmap('Blues_r', 128)
    
    #newcolors = np.vstack((top(np.linspace(0, 1, 128)),\
    #                       bottom(np.linspace(0, 1, 128))))
    newcolors = np.vstack((top_w,bottom_w))
    rb = ListedColormap(newcolors, name='RB')
    return rb

def white_yellow_green_blue():
    ylgnbu = cm.get_cmap('YlGnBu',256)
    newcolors = ylgnbu(np.linspace(0, 1, 256)) #256
    w = np.array([1, 1, 1, 1])
    newcolors[0:12, :] = w 
    wygb = ListedColormap(newcolors, name='WYGB')
    return wygb



