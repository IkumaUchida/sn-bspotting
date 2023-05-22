import glob
import re
import numpy as np

#for file sort
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]



def get_max_bbox_size(file_path,totalframecount):
    label_path = file_path + "/*.txt"
    max_bbox_size_array = np.zeros(totalframecount)
    
    for file in sorted(glob.iglob(label_path),key = natural_keys):
        max_bbox_size = 0
        with open(file) as f:
            frame = int(file.split('_')[-1].split('.txt')[0])
            for line in f:
                xywh = line.split(' ')
                #check label id 
                if(xywh[0] == '0'):
                    xywh.pop(0)
                    float_xywh = [float(item) for item in xywh]
                    if(float_xywh[2] * float_xywh[3] * 100 > max_bbox_size):
                        max_bbox_size = float_xywh[2] * float_xywh[3] * 100
            
            max_bbox_size_array[frame - 1] = max_bbox_size
    return max_bbox_size_array


def get_zoom_frames(max_bbox_size_array,threshold=1):
    zoom_frames = np.zeros(len(max_bbox_size_array))

    for i in range(len(max_bbox_size_array)):
        if(max_bbox_size_array[i] > threshold):       
            zoom_frames[i] = 1
        else:
            zoom_frames[i] = 0
    return zoom_frames

    
                
def main():
    #example
    totalframecount = 1188
    file_path = "/Users/kota/Desktop/soccer_net/labels_replay"
    threshold = 1
    
    max_bbox_size_array = get_max_bbox_size(file_path,totalframecount)
    zoomframes = get_zoom_frames(max_bbox_size_array,threshold)
    print(zoomframes)



if __name__ == '__main__':
    main()