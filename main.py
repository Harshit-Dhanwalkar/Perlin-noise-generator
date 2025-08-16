# main.py
import numpy as np
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
    shader=lit_with_shadows_shader,
)
# test_cube = Entity(model="cube", color=color.red, position=(0, -100, 0), scale=5)


# --- PLAYER SETUP ---
def find_safe_spawn_point():
    """Finds a safe spawn point for the player on land."""
    mid_x, mid_z = world_size // 2, world_size // 2
    height_scale = 10

    for offset in range(1, world_size // 4):
        x = mid_x + np.random.randint(
            max(-offset, -mid_x), min(offset, world_size - mid_x - 1)
        )
        z = mid_z + np.random.randint(
            max(-offset, -mid_z), min(offset, world_size - mid_z - 1)
        )

        if world_generator.grid[x, z] not in [0, 1]:  # Check if it's not water
            height = world_generator.noise[x, z] * height_scale * terrain.scale.y
            return (
                x * terrain.scale.x + terrain.x,
                height + 1.5,
                z * terrain.scale.z + terrain.z,
            )
    return (0, 150, 0)


start_position = find_safe_spawn_point()
player = FirstPersonController(
    position=start_position,
    mouse_sensitivity=Vec2(40, 40),
    origin_y=-0.5,
)

# --- LIGHTING & SKY ---
sun = DirectionalLight(y=20, rotation=(45, 120, 0), shadows=True)
sun.look_at(Vec3(0, -1, 0))
AmbientLight(color=color.rgba(50, 50, 50, 255))
Sky(texture="sky_default")

app.run()
