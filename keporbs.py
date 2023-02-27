import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import matplotlib
from astropy import units as u
from astropy.visualization import quantity_support
from astropy.constants import G
from scipy.spatial.transform import Rotation
from joblib import Parallel, delayed
quantity_support()
from astropy.io import ascii
from tqdm import tqdm
from ffmpy import FFmpeg
import os
import shutil 


num_frames=240
framerate=240
savename="Test_vid.mp4"
elev_start = 20
elev_end=0
azim_start=-50
azim_end= -60
use_darkmode=True
a_array = [100]*num_frames
e_array = [0.4]*num_frames
inc_array=[70]*num_frames
raan_array=[30]*num_frames
omega_array=[0]*num_frames

figsize=(10,10)
Vector_Scale=100
Plane_Size=150

fill_plane=True
show_axis=True
show_periapsis=True
show_apoapsis=True
show_longitudanal_nodes=True
show_h=True
show_RAAN=True
show_inclination=True

if azim_start<azim_end:
    az_arr=np.flip(np.arange(azim_end,azim_start,np.abs(azim_end-azim_start)/num_frames))
else:
    az_arr=np.arange(azim_start,azim_end,np.abs(azim_end-azim_start)/num_frames)

if elev_start<elev_end:
    el_arr=np.flip(np.arange(elev_end,elev_start,np.abs(elev_end-elev_start)/num_frames))
else:
    el_arr=np.arange(elev_start,elev_end,np.abs(elev_end-elev_start)/num_frames))

dark_red="#c1272d"
indigo="#0000a7" #- Indigo
yellow="#eecc16" #- Yellow
teal="#008176"# - Teal
light_grey="#b3b3b3" #- Light Gray
fontsize=20


def visualisation_plot(n):
    if use_darkmode:
        plt.style.use(['dark_background'])
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    a = a_array[n]
    e = e_array[n]
    b = a * np.sqrt(1 - e ** 2)
    r_p = a * (1 - e)
    r_a=a*(1+e)
    p = a * (1 - e ** 2)

    inclination = inc_arr[n]
    raan = raan_array[n]
    omega = omega_array[n]

    rot = R.from_euler("ZY", [raan, inclination], degrees=True)

    theta = np.arange(0, 2 * np.pi, step=0.01)
    phi = 0
    # https://math.stackexchange.com/a/819533
    r = a * (1 - e ** 2) / (1 - e * np.cos(theta - phi))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.zeros_like(x)
    r = np.vstack((x, y, z)).T
    rotated = rot.apply(r)

    vector_scale = Vector_Scale
    h_vec = np.zeros((2, 3), dtype=float)
    h_vec[1, 2] = vector_scale
    rot_vec = rot.apply(h_vec)

    periapsis = np.array(((0, 0, 0), (-r_p, 0, 0)))
    peri_rot = rot.apply(periapsis)

    apoapsis = np.array(((0, 0, 0), (r_a, 0, 0)))
    apo_rot = rot.apply(apoapsis)


    plane_size = Plane_Size
    plane = np.array(
        (
            (-plane_size, -plane_size, 0),
            (-plane_size, plane_size, 0),
            (plane_size, -plane_size, 0),
            (plane_size, plane_size, 0),
        )
    )
    inclination_range = np.arange(0, np.radians(inclination), step=0.01)
    arc = (
        vector_scale
        / 2
        * np.vstack(
            (
                np.sin(inclination_range),
                np.zeros_like(inclination_range),
                np.cos(inclination_range),
            )
        ).T
    )
    arc_rot = R.from_euler("Z", [raan], degrees=True)
    arc = arc_rot.apply(arc)
    arc_2 = (
        -r_p
        * np.vstack(
            (
                np.cos(inclination_range),
                np.zeros_like(inclination_range),
                -np.sin(inclination_range),
            )
        ).T
    )
    arc_2 = arc_rot.apply(arc_2)
    N_vec = np.cross([0, 0, 1], rot_vec[1, :])
    N = np.linalg.norm(N_vec)
    u_N = N_vec / N
    node_line = np.vstack(
        (
            np.linspace(-75, 75, 2),
            u_N[1] / u_N[0] * np.linspace(-75, 75, 2),
            np.zeros(2),
        )
    ).T

    nodes = np.array(((0, p, 0), (0, -p, 0)))
    nodes = rot.apply(nodes)

    raan_range = np.arange(0, np.radians(90 + raan), step=0.01)
    raan_arc = (
        vector_scale
        / 2
        * np.vstack((np.cos(raan_range), np.sin(raan_range), np.zeros_like(raan_range))).T
    )




    #Plot the body orbit

    ax.scatter3D(rotated[:,0],rotated[:,1],rotated[:,2],c=teal)
    #ax.plot(x,y)

    # fill the plane
    if fill_plane:
        ax.plot_trisurf(rotated[:, 0], rotated[:, 1], rotated[:, 2], alpha=0.2,color=teal)

        ax.plot_trisurf(plane[:, 0], plane[:, 1], plane[:, 2], alpha=0.2,color=light_grey)

    if show_axis:
        ax.plot3D([0,0],[0,0],[0,plane_size])
        ax.plot3D([0,0],[0,plane_size],[0,0])
        ax.plot3D([0,plane_size],[0,0],[0,0])
        #ax.text(-10, 20,155,"Z",fontsize=fontsize)
        #ax.text(0, 160,0,"Y",fontsize=fontsize)
        #ax.text(155, 0,0,"X",fontsize=fontsize)

    if show_periapsis:
        mid = peri_rot.shape[0] // 2
        ax.plot(xs=peri_rot[:, 0],ys=peri_rot[:, 1],zs=peri_rot[:, 2],color='#F4B58E')
        ax.text(peri_rot[1, 0]-40,peri_rot[1, 1],peri_rot[1, 2]+10,"Pericenter",fontsize=fontsize)

    if show_apoapsis:
        mid = apo_rot.shape[0] // 2
        ax.plot(xs=apo_rot[:, 0],ys=apo_rot[:, 1],zs=apo_rot[:, 2],color='#F4B58E')
        ax.text(apo_rot[1, 0]-40,apo_rot[1, 1],apo_rot[1, 2]+10,"Apocenter",fontsize=fontsize)

    if show_longitudanal_nodes:
        ax.plot(xs=node_line[:, 0],ys=node_line[:, 1],zs=node_line[:, 2],color=indigo)
        ax.scatter3D(nodes[1, 0],nodes[1, 1],nodes[1, 2],s=100,c=indigo)
        ax.scatter3D(nodes[0, 0],nodes[0, 1],nodes[0, 2],s=100,c=indigo)
        ax.text(nodes[1, 0],nodes[1, 1]-50,nodes[1, 2]-40,"Descending Node",fontsize=fontsize)
        ax.text(nodes[0, 0],nodes[0, 1]+10,nodes[0, 2]+40,"Ascending Node",fontsize=fontsize)


    if show_h:
        ax.quiver(rot_vec[0,0],rot_vec[0,1],rot_vec[0,2],rot_vec[1,0],rot_vec[1,1],rot_vec[1,2],arrow_length_ratio=0.15,linewidth=4,color='orange')
        ax.text(rot_vec[1,0]+5,rot_vec[1,1],rot_vec[1,2]-10,"h",fontsize=fontsize,color='orange')

    if show_inclination:
        mid = arc.shape[0] // 2
        label = [""] * arc.shape[0]
        label[mid] = "i"
        ax.scatter3D(xs=arc[:, 0],ys=arc[:, 1],zs=arc[:, 2],)
        ax.text(arc[mid, 0],arc[mid, 1],arc[mid, 2]+10,"i",fontsize=fontsize)


    if show_RAAN:
        mid = raan_arc.shape[0] // 2
        label = [""] * raan_arc.shape[0]
        label[mid] = "Ω"
        ax.scatter3D(xs=raan_arc[:, 0],ys=raan_arc[:, 1],zs=raan_arc[:, 2],color="#DEDAFA",s=2)
        ax.text(raan_arc[mid, 0]+5,raan_arc[mid, 1]+5,raan_arc[mid, 2]+0,label[mid],fontsize=fontsize)

    ax.set_xlim(-plane_size,plane_size)
    ax.set_ylim(-plane_size,plane_size)
    ax.set_zlim(-plane_size,plane_size)


    ax.scatter(0,0,0,s=500,c=yellow)
    #ax.view_init(0,-60)

    plt.axis('off')
    ax.view_init(el_arr[n], az_arr[n])

    plt.savefig(f"temp_anim_folder/frame_{n}.png")
    del fig
    del ax

if not os.path.isdir("temp_anim_folder"):
    os.mkdir("temp_anim_folder")

Parallel(n_jobs=6)(delayed(visualisation_plot)(num) for num in tqdm(range(num_frames)))

ffmpeg=(FFmpeg(inputs={"temp_anim_folder/frame_%d.png":None},outputs={savename:f"-framerate {framerate}"}))
ffmpeg.run()

shutil.rmtree("temp_anim_folder")
