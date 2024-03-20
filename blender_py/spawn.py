import bpy
                
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

location = [(0,0,0), (100,100,0)]
rotation = [(0.0, -0.0, 3.14), (0.0, -0.0, 3.14)]

for i in range(len(location)):
    
    print(location[i])
    spawn_objects(blend_filepath, location[i], rotation[i]) 
