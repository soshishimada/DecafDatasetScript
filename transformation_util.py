import torch,copy
from pytorch3d import transforms
import numpy as np

def rot_angle2rotmat(rot_angle:torch.Tensor):
  """ 
  Args:
      rot_angle (*,3):  
  return
      rotation 6d (*,3,3)
  """
  return  transforms.axis_angle_to_matrix(rot_angle )

def convert_R_T_to_RT4x4(rot_mat:torch.Tensor,transl:torch.Tensor)->torch.Tensor:
  """
  args:
    rot_mat (*,3,3)
    transl (*,3,)
  returns:
    RT matrix (*,4,4)
  """
  b=len(rot_mat)
  rot_mat = rot_mat.view(b,3,3)
  transl= transl.view(b,3,1)
  RT = torch.cat((rot_mat,transl),dim=2)
  bottom = torch.FloatTensor(
    [0,0,0,1]).view(1,1,4).expand(b,-1,-1).to(rot_mat.device)
  RT=torch.cat((RT,bottom),dim=1) 
  return RT


def apply_transform_np(data:np.ndarray,RT:np.ndarray)->np.ndarray:
    """
    a function to apply 3D transformation on a 3D data
    Args:
      data: n_vs,3
      RT: 4,4
    Return ( n_vs,3):
      data after transformation
    """ 
    n_vs,_=data.shape
    data = np.concatenate((data,np.ones((n_vs,1))),axis=-1) 
    return (np.dot(RT,data.T).T)[:,:-1]
  
def convert_canonical_deforms_into_cam_space(can_deforms,
                                             head_pose, 
                                              cam_transform
                                              ):
    """
    cannonical frame -> world frame -> camera frame
    
    can_deform (N,3): deformations in a FLAME canoincal space
    head_pose (4,4): head pose matrix 
    cam_transform (4,4): camera transformation matrix
    """ 
     
    head_pose[:3, -1] = 0 
    posed_deforms = apply_transform_np(data=can_deforms,
                                          RT=head_pose)
    ##### posed -> cam space ###
    cam_transform=copy.copy(cam_transform)
    cam_transform[:3, -1] = 0 
    cam_space_deforms = apply_transform_np(data=posed_deforms,
                                              RT=cam_transform)

    return cam_space_deforms #cam_space_deforms
  

def apply_transform_np(data:np.ndarray,RT:np.ndarray)->np.ndarray:
    """
    a function to apply 3D transformation on a 3D data
    Args:
      data: n_vs,3
      RT: 4,4
    Return ( n_vs,3):
      data after transformation
    """ 
    n_vs,_=data.shape
    data = np.concatenate((data,np.ones((n_vs,1))),axis=-1) 
    return (np.dot(RT,data.T).T)[:,:-1]
  
import cv2
def image_flipper_with_image_center(k,img):
    """
    A function to flip image vertically considering the image center.
    The image center needs to be taken into account when you flip the 3D geometry
    horizontally and want to obtain the corresponding flipped image.
    
    args:
      k (3,3): intrinsic matrix
      img (h,w,c): image
    """
    
     
    h,w,c=img.shape
    n =int( k[0][2] ) 
    
    if 2*n<w: 
      sub_img = img[:,:2*n]
      sub_img = cv2.flip(sub_img, 1)
      end_img = np.zeros((h, w-2*n, 3), dtype = np.uint8) #img[:,:w-2*n] 
    else:
      
      m=w-n 
      sub_img = img[:,w-2*m:]
      sub_img = cv2.flip(sub_img, 1)
      end_img = np.zeros((h,w-2*m, 3), dtype = np.uint8)
     
    return  np.concatenate((end_img,sub_img),axis=1)