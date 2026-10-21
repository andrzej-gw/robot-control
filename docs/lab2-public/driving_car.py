import mujoco
import matplotlib.pylab as plt
import math

def generate(px, py, cr):
    pz = 0
    
    crr = math.radians(cr)
    
    wrx = math.cos(crr)
    wry = math.sin(crr)
    
    
    return f"""
<?xml version="1.0" ?>
<mujoco>
    <visual>
        <global offwidth="1280" offheight="1280"/>
    </visual>
    <worldbody>
         <body name="floor" pos="0 0 -0.1">
            <geom size="2.0 2.0 0.02" rgba="0.2 0.2 0.2 1" type="box"/>
        </body>
        <body name="x_arrow" pos="0.5 0 0">
            <geom size="0.5 0.01 0.01" rgba="1 0 0 0.5" type="box"/>
        </body>
        <body name="y_arrow" pos="0 0.5 0">
            <geom size="0.01 0.5 0.01" rgba="0 1 0 0.5" type="box"/>
        </body>
        <body name="z_arrow" pos="0 0 0.5">
            <geom size="0.01 0.01 0.5" rgba="0 0 1 0.5" type="box"/>
        </body>
        
        <body name="car" pos="{0 + px} {0 + py} {0.1 + pz}" axisangle="0 0 1 {cr}">
            <geom size="0.2 0.1 0.02" rgba="1 1 1 0.9" type="box"/>
        </body>
        <body name="wheel_1" pos="{0.1*math.cos(crr)-0.1*math.sin(crr) + px} {0.1*math.sin(crr)+0.1*math.cos(crr) + py} {0.1 + pz}" axisangle="{wrx} {wry} 0 90">
            <geom size="0.07 0.01" rgba="1 1 1 0.9" type="cylinder"/>
        </body>
        <body name="wheel_2" pos="{-0.1*math.cos(crr)-0.1*math.sin(crr) + px} {-0.1*math.sin(crr)+0.1*math.cos(crr) + py} {0.1 + pz}" axisangle="{wrx} {wry} 0 90">
            <geom size="0.07 0.01" rgba="1 1 1 0.9" type="cylinder"/>
        </body>
        <body name="wheel_3" pos="{0.1*math.cos(crr)+0.1*math.sin(crr) + px} {0.1*math.sin(crr)-0.1*math.cos(crr) + py} {0.1 + pz}" axisangle="{wrx} {wry} 0 90">
            <geom size="0.07 0.01" rgba="1 1 1 0.9" type="cylinder"/>
        </body>
        <body name="wheel_4" pos="{-0.1*math.cos(crr)+0.1*math.sin(crr) + px} {-0.1*math.sin(crr)-0.1*math.cos(crr) + py} {0.1 + pz}" axisangle="{wrx} {wry} 0 90">
            <geom size="0.07 0.01" rgba="1 1 1 0.9" type="cylinder"/>
        </body>
        
        <body name="radar_1" pos="{0*math.cos(crr)-0.1*math.sin(crr) + px} {0*math.sin(crr)+0.1*math.cos(crr) + py} {0.2 + pz}" axisangle="{wrx} {wry} 0 -30">
            <geom size="0.01 0.01 0.1" rgba="1 1 1 1" type="box"/>
        </body>
        <body name="radar_2" pos="{0*math.cos(crr)-0.15*math.sin(crr) + px} {0*math.sin(crr)+0.15*math.cos(crr) + py} {0.29 + pz}" axisangle="{wrx} {wry} 0 -30">
            <geom size="0.03 0.01" rgba="1 0 0 1" type="cylinder"/>
        </body>
      
    </worldbody>
</mujoco>
"""
frames = 60

for i in range(frames):
    cr = 90/frames*i
    crr = math.radians(cr)
    px = math.sin(crr)
    py = -math.cos(crr)    
    px -= 1
    
    print(px, py)
    
    xml = generate(px, py, cr)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 1280, 1280)

    mujoco.mj_forward(model, data)
    renderer.update_scene(data)
    plt.imsave(f"frame_{i:03d}.png", renderer.render())
    renderer.close()
