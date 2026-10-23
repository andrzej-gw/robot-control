import mujoco
from PIL import Image
import cv2
import numpy as np

def find_colors(img, color, tolerance, debug = False):
    #  print(img.shape)
    result = []
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            for i in range(3):
                if not (color[i]-tolerance<img[x][y][i] and img[x][y][i]<color[i]+tolerance):
                    break
                if i == 2:
                    result.append((x, y))

    if debug:
        img2 = img.copy()
        for x, y in result:
            img2[x][y] = [255, 0, 255]

        combined = cv2.hconcat([img, img2])
            
        cv2.imshow(f"Detecion of {color}", combined)
        cv2.waitKey(0)          # wait for a key press
        cv2.destroyAllWindows()
    
    return result

def find_center(img, color, tolerance, debug = False):
    points = find_colors(img, color, tolerance, debug)
    average = np.mean(points, axis=0)  # [avg_x, avg_y]

    if debug:
        img2 = img.copy()
        x = int(average[0])
        y = int(average[1])
        img2[x][y] = [255, 0, 255]
        cv2.imshow("Center", img2)
        cv2.waitKey(0)          # wait for a key press
        cv2.destroyAllWindows()
    
    return average
    
# distance between two cameras
baseline = 0.1

sphere_size = 0.01
box1_size = 0.05
box2_size = 0.02

resolution = (1280, 1280)

img1 = cv2.imread('assets/left.png')
img2 = cv2.imread('assets/right.png')

#TODO: find the positions of the objects and reconstruct the images

# fovy: f * tan(fovy / 2) = image_height / 2
f = resolution[0]/2/np.tan(np.deg2rad(90/2))
print("f:", f)

# red sphere
print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\nred sphere:")
l = find_center(img1, (0, 0, 255*0.8), 125)
r = find_center(img2, (0, 0, 255*0.8), 125)

print("x-x':", l[1]-r[1])

Z=f*baseline/(l[1]-r[1])
print("Z:", Z)

pos_sphere = ((l[1]-resolution[0]/2)/f*Z, -(l[0]-resolution[1]/2)/f*Z, 1 - Z)

# green box
print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\ngreen box:")
l = find_center(img1, (0, 255*0.8, 0), 125)
r = find_center(img2, (0, 255*0.8, 0), 125)

print("x-x':", l[1]-r[1])

Z=f*baseline/(l[1]-r[1])
print("Z:", Z)

pos_box1 = ((l[1]-resolution[0]/2)/f*Z, -(l[0]-resolution[1]/2)/f*Z, 1 - Z-box1_size)

# blue box
print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\nblue box:")
l = find_center(img1, (255*0.8, 0, 0), 125)
r = find_center(img2, (255*0.8, 0, 0), 125)

print("x-x':", l[1]-r[1])

Z=f*baseline/(l[1]-r[1])
print("Z:", Z)

pos_box2 = ((l[1]-resolution[0]/2)/f*Z, -(l[0]-resolution[1]/2)/f*Z, 1 - Z-box2_size)

xml_string =f"""\
<mujoco model="simple_scene">
      <visual>
     <global offwidth="{resolution[0]}" offheight="{resolution[1]}"/>
  </visual>
    <asset>
        <texture name="plane_texture" type="2d" builtin="checker" rgb1="0.2 0.2 0.2" rgb2="0.3 0.3 0.3" width="512" height="512"/>
        <material name="plane_material" texture="plane_texture" texrepeat="5 5" reflectance="0."/>
    </asset>

    <worldbody>
        <geom type="plane" size="1 1 0.1" material="plane_material"/>

        <body pos="{pos_sphere[0]} {pos_sphere[1]} {pos_sphere[2]}">
            <geom type="sphere" size="{sphere_size}" rgba="1 0 0 1"/>
        </body>

        <body pos="{pos_box1[0]} {pos_box1[1]} {pos_box1[2]}">
            <geom type="box" size="0.05 0.05 0.05" rgba="0 1 0 1"/>
        </body>

        <body pos="{pos_box2[0]} {pos_box2[1]} {pos_box2[2]} ">
            <geom type="box" size="0.02 0.02 0.02" rgba="0 0 1 1"/>
        </body>

        <!-- Cameras -->
        <camera name="camera1" pos="0 0 1" fovy="90"/>
        <camera name="camera2" pos="{baseline} 0 1" fovy="90"/>
    </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml_string)
renderer = mujoco.Renderer(model, resolution[0], resolution[1])

data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

renderer.update_scene(data, camera="camera1")
img1_r = Image.fromarray(renderer.render())
img1_r.save("reconstruct_left.png")

renderer.update_scene(data, camera="camera2")
img2_r = Image.fromarray(renderer.render())
img2_r.save("reconstruct_right.png")
