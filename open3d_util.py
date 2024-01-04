import open3d as o3d
import numpy as np
def view_setup_known_cam(ext, visibility=True, img_size=1080,):
    view_setup_known_cam.index = -1 
    view_setup_known_cam.vis = o3d.visualization.Visualizer( )  
    vis = view_setup_known_cam.vis 
    #vis = o3d.visualization.Visualizer() 
    vis.create_window(width=img_size, height=img_size, visible=visibility)

    opt = vis.get_render_option()
    opt.mesh_show_back_face = True  # np.array([1,0,0])
    vis.create_window(width=img_size, height=img_size, visible=visibility)
    cam = vis.get_view_control().convert_to_pinhole_camera_parameters() 
    cam.extrinsic = ext
    return vis, cam
  
  

def get_mesh_from_vs_and_faces(vs, faces, colors=None):
    """vs:nx3 faces:kx3."""
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vs)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()

    if colors is None:
        mesh.paint_uniform_color(np.array([0.5, 0.5, 0.5]))
    else:
        mesh.paint_uniform_color(colors)
    return mesh

def change_mesh_color(mesh:o3d.geometry.TriangleMesh,c:np.ndarray):
    mesh.vertex_colors= o3d.utility.Vector3dVector(c)
    return mesh
def get_color_array_with_specified_ids(dim:int,
                                       t_idx:list,
                                       t_c:list,
                                       ori_color:np.ndarray=None) -> np.ndarray:
    """function that returns a color array for open3d mesh/pcd

    Args:
        dim (int): number of vertices 
        t_idx (list): target index
        t_c (3): target color

    Returns:
        np.ndarray: color array
    """
    if ori_color is not None:
        colors = ori_color
    else:
        colors = 0.5*np.ones((dim,3))
    colors[t_idx] = t_c
    
    return colors
def get_mesh_colored_by_idx(mesh:o3d.geometry.TriangleMesh,
                            t_idx:list,t_c:list,keep_ori_color=False)->o3d.geometry.TriangleMesh:
    """function that changes the mesh color with the specified id and color

    Args:
        mesh (int): number of vertices 
        t_idx (list): target index
        t_c (3): target color

    Returns:
        o3d.geometry.TriangleMesh: open3d mesh with the updated color
    """
    if keep_ori_color:
        ori_color = np.array(mesh.vertex_colors) 
    else:
        ori_color =None
    color = get_color_array_with_specified_ids(
                dim=len(np.array(mesh.vertices)),
                t_idx=t_idx,
                t_c=t_c,
                ori_color = ori_color)
    mesh = change_mesh_color(c=color, mesh=mesh)
    return mesh
  

def visualization_process_o3d( vis,meshes,  cam,
                              return_img = False,
                              save_img=False, 
                              file_save_path=None):  
    
    for mesh in meshes:
        vis.add_geometry(mesh)
        vis.update_geometry(mesh)    
    vis.get_view_control().convert_from_pinhole_camera_parameters(cam)
    vis.poll_events()
    vis.update_renderer()  
    if return_img:
        o3d_screenshot_mat =  vis.capture_screen_float_buffer(True) 
        image = (255.0 * np.asarray(o3d_screenshot_mat)).astype(np.uint8)
    else:
        image= None
    if save_img:
        vis.capture_screen_image(file_save_path,True)  
    for mesh in meshes:
        vis.remove_geometry(mesh)  
    return image

def static_visualization(meshes: list,
                         frame: bool = False,
                         frame_size=1.0) -> None:
    if frame:
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=frame_size, origin=[0, 0, 0])
        meshes += [mesh_frame]
    o3d.visualization.draw_geometries(meshes)
    return


def save_mesh_to_ply_o3d(path: str, mesh: o3d.geometry.TriangleMesh) -> None:
    o3d.io.write_triangle_mesh(path, mesh, write_ascii=True)
