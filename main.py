from ursina import *
from ursina.prefabs.first_person_controller import FirstPersonController
from ursina.shaders import lit_with_shadows_shader

from world_2d import WorldGenerator

app = Ursina()

# --- TERRAIN GENERATION ---
world_size = 256
world_generator = WorldGenerator(size=world_size, octaves=6, persistence=0.5)
world_generator.generate_terrain(
    seed=42,
    water_threshold=0.35,
    snow_threshold=0.2,
    rock_probability=0.06,
    dirt_probability=0.05,
)

vertices, triangles, colors, normals, uvs = world_generator.create_3d_mesh()

# Debug prints
print(f"Vertices shape: {vertices.shape}")
print(f"Triangles shape: {triangles.shape}")
print(f"Colors shape: {colors.shape}")
print(f"Normals shape: {normals.shape}")
print(f"UVs shape: {uvs.shape}")

# Create terrain entity
terrain = Entity(
    model=Mesh(
        vertices=vertices,
        triangles=triangles,
        colors=colors,
        normals=normals,
        uvs=uvs,
        mode="triangle",
    ),
    scale=(0.5, 20, 0.5),
    position=(-world_size / 2, 0, -world_size / 2),  # Center terrain
    collider="mesh",
    # texture=None,
    # double_sided=True,
)
# test_cube = Entity(model="cube", color=color.red, position=(0, -100, 0), scale=5)

# --- PLAYER SETUP ---
# start_x, start_z = world_size // 2, world_size // 2
# start_height = world_generator.noise[start_x, start_z] * 20 + 2
start_x = 50
# start_z = world_generator.noise[start_x, start_z] * 10
start_z = 50
start_height = 170

player = FirstPersonController(
    position=(start_x - world_size / 2, start_height, start_z - world_size / 2),
    # position=(start_x, start_z, start_height),
    # speed=10,
    # gravity=0.5,
    mouse_sensitivity=Vec2(40, 40),
    origin_y=-0.5,
)
player.camera_pivot.y = start_height

# --- LIGHTING & SKY ---
DirectionalLight(parent=scene, y=100, z=100, shadows=True)
Sky()

app.run()
