import math
import numpy as np

def read_align(path_to_align=None):

    no_of_frames = 75

    with open(path_to_align, 'r') as f:
        lines = f.readlines()	

    align = [(int(y[0])/1000, int(y[1])/1000, y[2]) for y in [x.strip().split(" ") for x in lines]]
    #print(align)
    
    words = ['sil']
    for start,end, word in align:
      words.extend([word]*int(math.floor(end) - math.floor(start)))

    time_indexed_alignment  = np.array(words)
    return time_indexed_alignment

#read_align(path_to_align)
