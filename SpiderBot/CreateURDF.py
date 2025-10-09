import numpy as np

# === Parameters ===
body_radius = 0.125 * 1.75
body_height = 0.025

leg_count = 6
hip_size = 0.025
upper_leg = 0.125
middle_leg = 0.175
lower_leg = 0.175
foot_radius = np.sqrt(2 * body_height**2) / 2

# Joint parameters 
# min, max, offset, effort (N*m), velocity (rad/s)
hip_info = [-55, 55, 0, 25, 4.0]
upper_info = [-75, 75, 0, 12, 5.0] # default -35
middle_info = [-90, 20, 0, 8, 6.0] # default 70
lower_info = [-90, 45, 0, 4, 8.0] # default 55

effort_mod = 1.0
velocity_mod = 1.0
damping = 0.5
friction = 0.01

# Density in kg/m^3
# PLA/plastic ~ 1200, Aluminum ~ 2700, Steel ~ 7850
body_density = 2400.0
leg_density = 1200.0
foot_density = 1200.0

# Turn first 3 limits to radians
hip_info = list(np.radians(hip_info[0:3])) + hip_info[3:]
upper_info = list(np.radians(upper_info[0:3])) + upper_info[3:]
middle_info = list(np.radians(middle_info[0:3])) + middle_info[3:]
lower_info = list(np.radians(lower_info[0:3])) + lower_info[3:]

def write_inertial(f, length=None, size=None, radius=None, height=None, density=1000):
    if length is not None and size is not None:
        # Box (rectangular prism)
        volume = length * size * size
        mass = density * volume
        ixx = (1/12) * mass * (size**2 + size**2)
        iyy = (1/12) * mass * (length**2 + size**2)
        izz = iyy
    elif radius is not None and height is not None:
        # Cylinder along X-axis
        volume = np.pi * radius**2 * height
        mass = density * volume
        ixx = 0.5 * mass * radius**2
        iyy = (1/12) * mass * (3 * radius**2 + height**2)
        izz = iyy
    elif radius is not None and height is None:
        # Sphere
        volume = (4/3) * np.pi * radius**3
        mass = density * volume
        ixx = (2/5) * mass * radius**2
        iyy = ixx
        izz = ixx
    else:
        raise ValueError("Need either (length, size) or (radius, height) for inertial calc.")

    f.write('    <inertial>\n')
    f.write(f'      <mass value="{mass:.6f}"/>\n')
    f.write(f'      <inertia ixx="{ixx:.6e}" iyy="{iyy:.6e}" izz="{izz:.6e}" ixy="0" ixz="0" iyz="0"/>\n')
    f.write('    </inertial>\n')


def write_geom(f, other_xml, shape_xml):
    f.write('    <visual>\n')
    f.write(f'      {other_xml}\n')
    f.write(f'      <geometry>{shape_xml}</geometry>\n')
    f.write('    </visual>\n')
    f.write('    <collision>\n')
    f.write(f'      {other_xml}\n')
    f.write(f'      <geometry>{shape_xml}</geometry>\n')
    f.write('    </collision>\n')

def write_leg(f, parent, child, i, parent_length, length, joint_info, axis="0 -1 0", use_i=True,
              x = None, y = 0, z = 0, angle = None):
    f.write(f'  <link name="{child}_{i}">\n')
    write_geom(
        f, 
        f'<origin xyz="{length/2} 0 0" rpy="0 0 0"/>',
        f'<box size="{length} 0.025 0.025"/>'
    )
    write_inertial(f, length=length, size=0.025, density=leg_density)
    f.write('  </link>\n')

    f.write(f'  <joint name="joint_{parent}_{child}_{i}" type="revolute">\n')
    if use_i:
        f.write(f'    <parent link="{parent}_{i}"/>\n')
    else:
        f.write(f'    <parent link="{parent}"/>\n')
    f.write(f'    <child link="{child}_{i}"/>\n')
    if x is None:
        x = parent_length
    if angle is None:
        angle = joint_info[2]
    f.write(f'    <origin xyz="{x} {y} {z}" rpy="0 0 {angle}"/>\n')
    f.write(f'    <axis xyz="{axis}"/>\n')
    f.write(f'    <limit lower="{joint_info[0]}" upper="{joint_info[1]}" effort="{joint_info[3]}" velocity="{joint_info[4]}"/>\n')
    f.write(f'    <dynamics damping="{damping}" friction="{friction}"/>\n')
    f.write('  </joint>\n')
    
def write_foot(f, parent, child, i, parent_length, radius):
    f.write(f'  <link name="{child}_{i}">\n')
    write_geom(f, '', f'<sphere radius="{radius}"/>')
    write_inertial(f, radius=radius, density=foot_density)
    f.write('  </link>\n')

    f.write(f'  <joint name="joint_{parent}_foot_{i}" type="revolute">\n')
    f.write(f'    <parent link="{parent}_{i}"/>\n')
    f.write(f'    <child link="{child}_{i}"/>\n')
    f.write(f'    <origin xyz="{parent_length} 0 0" rpy="0 0 0"/>\n')
    f.write(f'    <axis xyz="0 1 0"/>\n')
    f.write(f'    <limit lower="0" upper="0" effort="0" velocity="0"/>\n')
    f.write('  </joint>\n')

with open("spider.urdf", "w") as f:
    f.write('<?xml version="1.0"?>\n')
    f.write('<robot name="blocky_spider">\n')

    # === Body ===
    f.write('  <link name="body">\n')
    write_geom(f, '', f'<cylinder radius="{body_radius}" length="{body_height}"/>')
    write_inertial(f, radius=body_radius, height=body_height, density=body_density)
    f.write('  </link>\n')

    for i in range(leg_count):
        angle = i * (2*np.pi/leg_count)

        write_leg(f, "body", "leg_hip", i, body_radius, hip_size, hip_info, 
                  axis="0 0 1", use_i=False,
                  x = body_radius * np.cos(angle),
                  y = body_radius * np.sin(angle),
                  angle = angle + hip_info[2]
                  )
        write_leg(f, "leg_hip", 'leg_upper', i, hip_size, upper_leg, upper_info)
        write_leg(f, "leg_upper", 'leg_middle', i, upper_leg, middle_leg, middle_info)
        write_leg(f, "leg_middle", 'leg_lower', i, middle_leg, lower_leg, lower_info)

        write_foot(f, "leg_lower", "leg_foot", i, lower_leg, foot_radius)

    f.write('</robot>\n')

