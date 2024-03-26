import bpy
import json
import os
import numpy as np
import mathutils
from typing import List, Tuple     
     
def add_sun():
    
    sun_data = bpy.data.lights.new(name='NewSun', type='SUN')
    sun_object = bpy.data.objects.new('NewSunObject', sun_data)
    bpy.context.collection.objects.link(sun_object)
    sun_data.energy = 1.0
    sun_data.specular_factor = 1.2
    sun_data.angle = 0.523599
    sun_object.location = (0.0, 10, 10)
    sun_object.rotation_euler = (0.0, 0.0, 0.0)
    
def add_main_car_and_camera(car):
     
    camera_data = bpy.data.cameras.new(name='NewCamera')
    camera_object = bpy.data.objects.new('NewCameraObject', camera_data)
    bpy.context.collection.objects.link(camera_object)
    bpy.context.scene.camera = camera_object
    camera_object.location = (0.0, 1, 1.53)
    camera_object.rotation_euler = (1.57, 0.0, 0.0)
    camera_object.data.lens = 40
    
    # tintin's car
    spawn_objects(car, None, (0,0,0), (0,0,3.14), (0.022, 0.022, 0.022)) 

    camera_location = (0.0, 0, 1.53)
    camera_rotation = (1.57, 0.0, 0.0)

    if "NewCameraObject" in bpy.data.objects:
        camera = bpy.data.objects["NewCameraObject"]

    camera.location = camera_location
    camera.rotation_euler = camera_rotation


def calculate_segments(lines: List[List[Tuple[float, float, float]]], n: int) -> List[List[List[float]]]:
    all_segments = []
    
    # Function to calculate distance between two points
    def distance(p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)
    
    # Iterate over each line
    for points in lines:
        segments = []
        # Iterate over each pair of points in the line
        for i in range(len(points) - 1):
            p1, p2 = points[i], points[i + 1]
            
            # Calculate the distance between the current pair of points
            dist = distance(p1, p2)
            
            # Calculate the total number of segments and gaps
            total_parts = 2 * n
            part_length = dist / total_parts
            
            # Calculate the direction vector from p1 to p2
            direction = [(p2[0] - p1[0]) / dist, (p2[1] - p1[1]) / dist, (p2[2] - p1[2]) / dist]
            
            # Generate segments
            for segment in range(n):
                start = [p1[0] + direction[0] * part_length * (2 * segment), 
                         p1[1] + direction[1] * part_length * (2 * segment),
                         p1[2] + direction[2] * part_length * (2 * segment)]
                
                end = [start[0] + direction[0] * part_length, 
                       start[1] + direction[1] * part_length,
                       start[2] + direction[2] * part_length]
                
                segments.append([start, end])
        
        all_segments.append(segments)
            
    return all_segments

def add_lane(line_data, type_list):

    curves = []  # Store the created curves here

    # Iterate over each line's data to create a new curve
    for index, line in enumerate(line_data):
        # Create a new curve data object
        curve_data = bpy.data.curves.new(f'BezierCurve_{index}', type='CURVE')
        curve_data.dimensions = '3D'
        
        # Create a new object with the curve data
        curve_object = bpy.data.objects.new(f'BezierCurveObject_{index}', curve_data)
        scene = bpy.context.scene
        scene.collection.objects.link(curve_object)
        
        # Create a new spline in the curve
        spline = curve_data.splines.new(type='BEZIER')
        spline.bezier_points.add(len(line) - 1)  # Number of points per curve
        
        # Assign the coordinates to the spline's points
        for point_index, coord in enumerate(line):
            point = spline.bezier_points[point_index]
            point.co = mathutils.Vector(coord)  # Assign the tuple to the point
            point.handle_right_type = 'AUTO'
            point.handle_left_type = 'AUTO'
        
        curve_object.data.fill_mode = 'FULL'
        curve_object.data.bevel_depth = 0.1
        curve_object.scale = (1, 1, 0.1)
        
        # print(index, type_list[index])
        
        if type_list[index] == 'divider-line':
            
            # Create a new material
            material_color = bpy.data.materials.new(name="Yellow")

            # Set the material's diffuse color
            material_color.diffuse_color = (1.0, 1.0, 0.0, 1.0)  # RGB and alpha (red, in this case)
            
            # Assign the material to the cylinder
            if curve_object.data.materials:
                # Assign to 1st material slot if it exists
                curve_object.data.materials[0] = material_color
            else:
                # No slots
                curve_object.data.materials.append(material_color)
        
        curves.append(curve_object)  # Store the created curve object

        if type_list[index] == 'solid-line':
            
            # Create a new material
            material_color = bpy.data.materials.new(name="Yellow")

            # Set the material's diffuse color
            material_color.diffuse_color = (0.0, 0.0, 1.0, 1.0)  # RGB and alpha (red, in this case)
            
            # Assign the material to the cylinder
            if curve_object.data.materials:
                # Assign to 1st material slot if it exists
                curve_object.data.materials[0] = material_color
            else:
                # No slots
                curve_object.data.materials.append(material_color)
        
        curves.append(curve_object)  # Store the created curve object

                   
def texture(name, image_path, material_name):

    image = bpy.data.images.load(image_path)
    obj = bpy.data.objects[name]
    material = bpy.data.materials.new(material_name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    shader = nodes.new(type='ShaderNodeBsdfPrincipled')
    texture_node = nodes.new('ShaderNodeTexImage')
    texture_node.image = image
    links = material.node_tree.links
    link = links.new(texture_node.outputs['Color'], shader.inputs['Base Color'])
    material_output = nodes.new(type='ShaderNodeOutputMaterial')
    links.new(shader.outputs['BSDF'], material_output.inputs['Surface'])

    if obj.data.materials:
        obj.data.materials[0] = material
    else:
        obj.data.materials.append(material)                
                
def spawn_objects(filepath, type, location, rotation, scale):
    with bpy.data.libraries.load(filepath) as (data_from, data_to):
        
        data_to.objects = data_from.objects
    spawned_objects = []
    # print(data_to.objects)
        
    for obj in data_to.objects:
    
        
        if obj.name.startswith('Sun') or obj.name.startswith('Light') or obj.name.startswith('Camera'):
            # print("deleting", obj.name)
            bpy.data.objects.remove(obj, do_unlink=True)
            
        
        #if obj.parent is None:  # Only consider objects without parents (top-level objects)
        else:

            obj.location = location
            obj.rotation_euler = rotation
            obj.scale = scale
            bpy.context.collection.objects.link(obj)  # Corrected here
            spawned_objects.append(obj)            
            
            if type == 'stop sign':
                image_path = "/home/ashd/WPI Spring 2024/Computer Vision/Einstein_vision/RBE549-EinsteinVison/P3Data/Assets/StopSignImage.png"
                material_name = "StopSignMaterial"
                texture(obj.name, image_path, material_name)
                
            if type == 'go':
                
                location = obj.location
                x, y, z = obj.location.x + 0.05, obj.location.y - 0.2, obj.location.z + 0.3
                bpy.ops.mesh.primitive_cylinder_add(radius=0.2, depth=0.2, vertices=32,
                                    location=(x, y, z), rotation=(1.57,0,0))
                                
                cylinder = bpy.context.active_object
                cylinder.name = "CustomCylinder"
                cylinder.data.name = "CustomCylinderMesh"
                
                # Create a new material
                material_color = bpy.data.materials.new(name="Green")

                # Set the material's diffuse color
                material_color.diffuse_color = (0.0, 1.0, 0.0, 1.0)  # RGB and alpha (red, in this case)
                
                # Assign the material to the cylinder
                if cylinder.data.materials:
                    # Assign to 1st material slot if it exists
                    cylinder.data.materials[0] = material_color
                else:
                    # No slots
                    cylinder.data.materials.append(material_color)
                
            elif type == 'stop':
                
                location = obj.location
                x, y, z = obj.location.x + 0.05, obj.location.y - 0.2, obj.location.z + 1.4
                bpy.ops.mesh.primitive_cylinder_add(radius=0.2, depth=0.2, vertices=32,
                                    location=(x, y, z), rotation=(1.57,0,0))
                                    
                cylinder = bpy.context.active_object
                cylinder.name = "CustomCylinder"
                cylinder.data.name = "CustomCylinderMesh"
                
                # Create a new material
                material_color = bpy.data.materials.new(name="Red")

                # Set the material's diffuse color
                material_color.diffuse_color = (1.0, 0.0, 0.0, 1.0)  # RGB and alpha (red, in this case)
                        
                # Assign the material to the cylinder
                if cylinder.data.materials:
                    # Assign to 1st material slot if it exists
                    cylinder.data.materials[0] = material_color
                else:
                    # No slots
                    cylinder.data.materials.append(material_color)
                
            bpy.ops.object.select_all(action='DESELECT')
                    
# Delete all objects except for the camera
for obj in bpy.context.scene.objects:
    # if obj.type != 'CAMERA':
    bpy.data.objects.remove(obj, do_unlink=True)

car = "/home/ashd/blender-4.0.2-linux-x64/Assets/Vehicles/SedanAndHatchback.blend"
pedestrian = "/home/ashd/blender-4.0.2-linux-x64/Assets/Pedestrain.blend"
go = "/home/ashd/blender-4.0.2-linux-x64/Assets/TrafficSignal.blend"
stop = "/home/ashd/blender-4.0.2-linux-x64/Assets/TrafficSignal.blend"
stop_sign = "/home/ashd/blender-4.0.2-linux-x64/Assets/StopSign.blend"
truck = "/home/ashd/WPI Spring 2024/Computer Vision/Einstein_vision/RBE549-EinsteinVison/P3Data/Assets/Vehicles/Truck.blend"

add_sun()
add_main_car_and_camera(car)

# Map the strings to the corresponding variables or paths
type_paths = {
    'car': car, 'person': pedestrian, 'go': go, 'stop': stop, 'stop sign': stop_sign, 'truck': car
}

scene = '1'
frame = 1820
# print(frame)
# Path to your JSON file
file_path = '/home/ashd/WPI Spring 2024/Computer Vision/Einstein_vision/RBE549-EinsteinVison/blender_py/json_files/scene' + scene + '/scene.json'

# Open the JSON file and load its conten
with open(file_path, 'r') as file:
    data = json.load(file)

# Initialize a list to store details for each frame
frames = []
types = []
positions= []
rotations = []
scales = []

# print(data[frame])

frame_data = data[frame//10]

# for frame_data in data[frame]:
    # Iterate through each object in the frame
for obj in frame_data['objects']:
    types.append(obj['type'])
    positions.append(obj['position'])
    rotations.append(obj['rotation'])
    scales.append(obj['scale'])

# one frame at a time
# break

for i in range(len(positions)):
    
    x, y, z = (positions[i]['x'], positions[i]['y'], positions[i]['z'])
    rot_mat = 4*np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    new = rot_mat @ np.array([x, y, z]).T
    new_x, new_y, new_z = new[0], new[1], -new[2]
    phi, theta, psi = (rotations[i]['x'], rotations[i]['y'], rotations[i]['z'])
    scale_x, scale_y, scale_z = (scales[i]['x'], scales[i]['y'], scales[i]['z'])

    if types[i] == "traffic_light" or types[i] == "traffic light":
        
        continue
    
    object_path = type_paths[types[i]]

    if object_path == '/home/ashd/blender-4.0.2-linux-x64/Assets/Vehicles/SedanAndHatchback.blend':
        phi, theta, psi = (0, 0, 3.14)
        scale_x, scale_y, scale_z = (0.015, 0.015, 0.015)
        new_z = 0
    if object_path == '/home/ashd/blender-4.0.2-linux-x64/Assets/StopSign.blend':
        phi, theta, psi = (-4.71, 0, -1.57)
        scale_x, scale_y, scale_z = (0.5, 0.5, 0.5)
        new_z = 0
    if object_path == '/home/ashd/blender-4.0.2-linux-x64/Assets/Pedestrain.blend':
        phi, theta, psi = (1.57, 0, 3.14)
        scale_x, scale_y, scale_z = (0.015, 0.015, 0.015)
    if object_path == '/home/ashd/blender-4.0.2-linux-x64/Assets/TrafficSignal.blend':
        phi, theta, psi = (1.57, 0, -1.57)
        scale_x, scale_y, scale_z = (0.5, 0.5, 0.5)
    
    spawn_objects(object_path, types[i], (new_x, new_y, new_z), (phi, theta, psi), (scale_x, scale_y, scale_z)) 

# lanes

# Path to your JSON file
file_path = '/home/ashd/WPI Spring 2024/Computer Vision/Einstein_vision/RBE549-EinsteinVison/blender_py/json_files/scene' + scene + '/scene_lanes.json'

def process_points(json_file_path):
    # Load the JSON file
    with open(json_file_path) as file:
        data = json.load(file)

    frame_data = data[frame//10]
    
    # Extract and process the points
    points_list = []
    type_list = []
    # for frame in data:
    for lane in frame_data["lanes"]:
    
        rot_mat = 4*np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
        processed_points = [rot_mat @ np.array([point[0], 0, point[2]]) for point in lane["points"]]
        points_list.append(processed_points)
        type_list.append(lane["type"])

    return points_list, type_list



# Process points
line_data, type_list = process_points(file_path)

#line_data = [[(-3, 0, 0),(-3, 7, 0), (-2, 15, 0)],
#             [(3, 0, 0),(3, 7, 0), (2, 15, 0)],
#             [(0, 0, 0),(0, 5, 0), (0, 10, 0)]]

segments = []

# n = 20 # Specify the number of segments you want
# segments_output = calculate_segments(line_data, n)

# print(line_data)


add_lane(line_data, type_list)

#for i in range(len(line_data)):
#    
#    add_lane(line_data[i])


print("all worked")
print("---------------")

#bpy.ops.render.render(write_still=True)
#print(f"Rendered image {i:06d}.png")
#image_name = f"{i:06d}.png"
#image_filepath = os.path.join("/home/ashd/WPI Spring 2024/Computer Vision/Einstein_vision/RBE549-EinsteinVison/blender_py/rendered_images", image_name)
#bpy.data.images['Render Result'].save_re,nder(image_filepath)