# %%

from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

IMAGE_FOLDER = "./images"
LEAF_IMG = os.path.join(IMAGE_FOLDER, "leaf.jpg")
EIN_IMG = os.path.join(IMAGE_FOLDER, "ein.jpg")
BLACK_IMG = os.path.join(IMAGE_FOLDER, "black.jpg")

def show_img(img, name=None):
    np_img = np.asarray(img)
    if name is not None:
        print(name)
    if len(np_img.shape) == 2 or np_img.shape[2] == 1:
        plt.imshow(np_img, cmap="gray")
    elif np_img.shape[2] == 3:
        plt.imshow(np_img)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    print(f"Shape: {np_img.shape}", end="\n\n")

FILTERS = {
    "sobel": {
        "x": np.asarray([[1,2,1],
                        [0,0,0],
                        [-1,-2,-1]]),
        "y": np.asarray([[-1,0,1],
                        [-2,0,2],
                        [-1,0,1]])
    },
    "prewitt": {
        "x": np.asarray([[1,1,1],
                        [0,0,0],
                        [-1,-1,-1]]),
        "y": np.asarray([[-1,0,1],
                        [-1,0,1],
                        [-1,0,1]])
    },
    "robinson": {
        "n": np.asarray([[1,0,1],
                        [-2,0,2],
                        [-1,0,1]]),
        "nw": np.asarray([[0,1,2],
                        [-1,0,1],
                        [-2,-1,0]]),
        "w": np.asarray([[1,2,1],
                        [0,0,0],
                        [-1,-2,-1]]),
        "sw": np.asarray([[2,1,0],
                        [1,0,-1],
                        [0,-1,-2]]),
        "s": np.asarray([[1,0,-1],
                        [2,0,-2],
                        [1,0,-1]]),
        "se": np.asarray([[0,-1,-2],
                        [1,0,-11],
                        [2,1,0]]),
        "e": np.asarray([[-1,-2,-1],
                        [0,0,0],
                        [1,2,1]]),
        "ne": np.asarray([[-2,-1,0],
                        [-1,0,1],
                        [0,1,2]])   
    },
    "laplacian": {
        "pos": np.asarray([[0,1,0],
                        [1,-4,1],
                        [0,1,0]]),
        "neg": np.asarray([[0,-1,0],
                        [-1,4,-1],
                        [0,-1,0]])
    }
}

def get_grid(x, y, homogenous=False):
    coords = np.indices((x, y)).reshape(2, -1)
    return np.vstack((coords, np.ones(coords.shape[1]))) if homogenous else coords

# Define Transformations
def get_rotation(angle):
    angle = np.radians(angle)
    return np.array([ 
        [np.cos(angle), -np.sin(angle), 0], 
        [np.sin(angle),  np.cos(angle), 0], 
        [0, 0, 1]])
def get_translation(tx, ty):
    return np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ])
def get_scale(s):
    return np.array([
        [s, 0, 0],
        [0, s, 0],
        [0, 0, 1]
    ])


# %%

# Load image
img = cv2.imread(EIN_IMG)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
show_img(img, "Original Image")
img = np.asarray(img)

# %%

############# Image transformation

height, width = img.shape[:2]
print(height)
print(width)

# %%

# Grid to represent image coordinate I(x, y)
coords = get_grid(width, height, True).astype(np.int)
x, y = coords[0], coords[1]
print(coords)
# %%

# Transformation matrix which rotate 45 degree
R = get_rotation(45)
R 

# %%

warp_coords = np.round(R@coords).astype(np.int)
warp_coords

# %%
warp_x, warp_y = warp_coords[0, :], warp_coords[1, :]

# %%

# Get pixels within image boundary
indices = np.where((warp_x >= 0) & (warp_x < width) &
                   (warp_y >= 0) & (warp_y < height))

filtered_warp_x, filtered_warp_y = warp_x[indices], warp_y[indices]
filtered_x, filtered_y = x[indices], y[indices]

# Copy values to new image I'(x, y)
warp_img = np.zeros_like(img).astype(np.int)
warp_img[filtered_warp_y, filtered_warp_x] = img[filtered_y, filtered_x]
show_img(warp_img)


# %%

#prewitt
kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
img_prewittx = cv2.filter2D(gray, -1, kernelx)
img_prewitty = cv2.filter2D(gray, -1, kernely)

show_img(img_prewittx, "Prewitt X")
show_img(img_prewitty, "Prewitt Y")
show_img(img_prewittx + img_prewitty, "Prewitt")

# %%

#sobel
kernelx = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
kernely = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
img_sobelx = cv2.filter2D(gray, -1, kernelx)
img_sobely = cv2.filter2D(gray, -1, kernely)
# img_sobelx = cv2.Sobel(gray,cv2.CV_8U,1,0,ksize=5)
# img_sobely = cv2.Sobel(gray,cv2.CV_8U,0,1,ksize=5)
img_sobel = img_sobelx + img_sobely

show_img(img_sobelx, "Sobel X")
show_img(img_sobely, "Sobel Y")
show_img(img_sobel, "Sobel")

# %%

#canny
img_canny = cv2.Canny(img, 100, 200)
show_img(img_canny, "Canny")




