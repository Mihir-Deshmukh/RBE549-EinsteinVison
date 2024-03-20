import bpy
import json
import os
                
def spawn_objects(filepath, location, rotation):
    with bpy.data.libraries.load(filepath) as (data_from, data_to):
        data_to.objects = data_from.objects
    spawned_objects = []
    for obj in data_to.objects:
        if obj.parent is None:  # Only consider objects without parents (top-level objects)
            new_obj = obj.copy()
            new_obj.data = obj.data.copy()
            new_obj.animation_data_clear()
            new_obj.location = location
            new_obj.rotation_euler = rotation
            new_obj.scale = (0.02, 0.02, 0.02)
            bpy.context.collection.objects.link(new_obj)  # Corrected here
            spawned_objects.append(new_obj)
            for child_obj in obj.children:  # Iterate through child objects and link them
                new_child_obj = child_obj.copy()
                new_child_obj.data = child_obj.data.copy()
                new_child_obj.animation_data_clear()
                new_child_obj.parent = new_obj
                bpy.context.collection.objects.link(new_child_obj)  # Corrected here
                spawned_objects.append(new_child_obj)

# Delete all objects except for the camera
for obj in bpy.context.scene.objects:
    if obj.type != 'CAMERA':
        bpy.data.objects.remove(obj, do_unlink=True)

blend_filepath = "/home/ashd/blender-4.0.2-linux-x64/Assets/Vehicles/SedanAndHatchback.blend"
# blend_filepath = "/home/ashd/blender-4.0.2-linux-x64/Assets/Vehicles/Truck.blend"

# Path to your JSON file
file_path = '/home/ashd/WPI Spring 2024/Computer Vision/Einstein_vision/RBE549-EinsteinVison/spawn.json'

# Open the JSON file and load its content
with open(file_path, 'r') as file:
    data = json.load(file)

# Initialize a list to store details for each frame
frames = []
types = []
positions= []
rotations = []
scales = []

# tintin's car
spawn_objects(blend_filepath, (0,0,0), (0,0,3.14)) 

camera_location = (0.0, 0.2, 1.4)
camera_rotation = (1.57, 0.0, 0.0)

if "Camera" in bpy.data.objects:
    camera = bpy.data.objects["Camera"]

camera.location = camera_location
camera.rotation_euler = camera_rotation     

for frame_data in data:
    # Iterate through each object in the frame
    for obj in frame_data['objects']:
        types.append(obj['type'])
        positions.append(obj['position'])
        rotations.append(obj['rotation'])
        scales.append(obj['scale'])
    
    break
        
# print(len(frame_data))

for i in range(len(positions)):
    
    x, y, z = (positions[i]['x'], positions[i]['y'], 0)
    phi, theta, psi = (rotations[i]['x'], rotations[i]['y'], rotations[i]['z'])

    spawn_objects(blend_filepath, (x, y, z), (phi, theta, psi)) 
    
bpy.ops.render.render(write_still=True)
print(f"Rendered image {i:06d}.png")
image_name = f"{i:06d}.png"
image_filepath = os.path.join("/home/ashd/WPI Spring 2024/Computer Vision/Einstein_vision/RBE549-EinsteinVison/blender_py/rendered_images", image_name)
bpy.data.images['Render Result'].save_render(image_filepath)