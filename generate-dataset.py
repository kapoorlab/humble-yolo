from PIL import Image, ImageDraw
import random
import string
import csv
import os
import numpy as np

def one_hot(x, length):
    return [1 if x==i else 0 for i in range(length)]

def get_word(c):
    words = ["Trump Lost", "Biden Won", "Haris Won", "none"]
    return (words[c], one_hot(c,len(words)))

cell_w = 32
cell_h = 32
grid_w = 2
grid_h = 2
img_w=grid_w*cell_w
img_h=grid_h*cell_h
for j in range(0,5000):
        img = Image.new('RGB', (grid_w*cell_w,grid_h*cell_h))
        d = ImageDraw.Draw(img)
    
        Event_data = []
        for row in range(grid_w):
            for col in range(grid_h):
                
                (digits, cat) = get_word(random.randint(0,3))

                width = len(digits)*6
                
                if(digits=='none'):
                    Event_data.append([cat[0],cat[1],cat[2],cat[3], (col*cell_w+cell_w/2)/cell_w, (row*cell_h+cell_h/2)/cell_h, cell_w/img_w, cell_h/img_h,0]) # confidence of object
                    print("None", cat[0],cat[1],cat[2], (col*cell_w+cell_w/2)/cell_w, (row*cell_h+cell_h/2)/cell_h, cell_w/img_w, cell_h/img_h,0)
                    
                else:
                    x = random.randrange(col*cell_w, (col+1)*cell_w)
                    y = random.randrange(row*cell_w, min(67, (row+1)*cell_h))
                    
                    d.text((x-width/2, y-10/2), digits, fill=(255,255,255))
                    Event_data.append([cat[0],cat[1],cat[2],cat[3], x/cell_w, y/cell_h, width/img_w, 10/img_h, 1]) # confidence of object
                    
                    print("Objt", (col,row), (x/cell_w, y/cell_h, width/img_w, 10/img_h), 1)
        FileName = 'Labels' + "/" + str(j)  + ".txt"             
        if os.path.exists(FileName):
                os.remove(FileName)            
        writer = csv.writer(open(FileName, "a"))
        writer.writerows(Event_data)
        img.save('Images/%d.PNG' % j)


