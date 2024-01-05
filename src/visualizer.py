import os,argparse,torch,cv2,mano 
import numpy as np  
import FLAME_util,util,system_util,tracking_util
import open3d_util as o3du
import transformation_util as tranu 
 
def clean_deform_batch( deforms):
  """
  Input: deforms (*,5023,3)
  output: valid deforms (*,5023,3)
  """
  deforms[:, ignore_idx] = 0
  return deforms

def main(args,mode,sub):
  
  ### video loading ###
  vid_path = args.dataset_path + '/'+mode+'/videos/'+sub+'/'
  seg_path = args.dataset_path + '/'+mode+'/segmentations/'+sub+'/'
  vid_names = os.listdir(vid_path)
  vid_names.sort()
   
  vid_name = [x for x in vid_names if args.cam == x[:-4]][0]
  vid = cv2.VideoCapture(vid_path+vid_name)
  n_frames=int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
   
  if args.vis2d:
    ### segmentatoin mask loading ###
    if args.vis_mask:
      seg = cv2.VideoCapture(seg_path+vid_name)
       
      assert vid.get(cv2.CAP_PROP_FRAME_COUNT) == \
            seg.get(cv2.CAP_PROP_FRAME_COUNT) == \
            n_frames
    else:
      seg= None
    ### keypoints loading ###
    head_keys = np.load(args.dataset_path + '/'+mode+'/head_keys/'+sub+'/'+vid_name[:-4]+".npy" )
    rh_keys = np.load(args.dataset_path + '/'+mode+'/right_hand_keys/'+sub+'/'+vid_name[:-4]+".npy" )
    
    assert n_frames == \
            head_keys.shape[0] == \
            rh_keys.shape[0]
   
  if args.vis3d:
    
    ### contacts, deformation, parameters, camera loading ###
    contacts_head = np.load(args.dataset_path + '/'+mode+'/contacts/'+sub+'/contacts_head.npy')
    contacts_rh = np.load(args.dataset_path + '/'+mode+'/contacts/'+sub+'/contacts_rh.npy')  
    deformations = np.load(args.dataset_path + '/'+mode+'/deformations/'+sub+'/deforms.npy')
    params=system_util.pickle_loader(args.dataset_path + '/'+mode+'/params/'+sub+'/params_smoothed.pkl')
    cam_params= system_util.pickle_loader(args.dataset_path + '/'+mode+'/cameras/'+sub+'/cameras.pkl')
    Ps= np.matmul(cam_params[args.cam]['intrinsic'],
                  cam_params[args.cam]['extrinsic'][:-1])
    Ps= torch.FloatTensor(Ps).to(args.device)
    Ps_batch = Ps.clone().view(1,  1, 3, 4).expand(n_frames, -1, -1, -1) 
    RT_np=np.array(cam_params[args.cam]['extrinsic']) 
    
    parsed_parms = util.params_parser(params)
    head_rot_mat = tranu.rot_angle2rotmat(parsed_parms['head']['global_orient']) 
    head_poses = tranu.convert_R_T_to_RT4x4(head_rot_mat,
                                            parsed_parms['head']['transl'])
    rh_con_idx =\
            [np.nonzero(contacts_rh[i])[0] for i in range(len(contacts_rh))]
     
    head_con_idx = [np.nonzero(contacts_head[i])[0]
                    for i in range(len(contacts_head))]
    assert n_frames == \
          contacts_head.shape[0] == \
          contacts_rh.shape[0] == \
          deformations.shape[0] == \
          len(params.keys())
      
    ### initialize and run FLAME model ###
    flame_model = FLAME_util.FLAME( 
                         flame_model_path,
                         flame_landmark_path,
                          ).to(args.device)
    flame_faces = FLAME_util.get_FLAME_faces(flame_model_path)
     
     
     
    [head_vs, landmarks3d, head_keys_proj] = \
      FLAME_util.flame_forwarding(
      flame_model=flame_model,
      head_shape_params=parsed_parms['head']['shape_params'].to(args.device),
      head_expression_params=parsed_parms['head']['expression_params'].to(args.device),
      head_pose_params=parsed_parms['head']['pose_params'].to(args.device),
      head_rotation= parsed_parms['head']['global_orient'].to(args.device),
      head_transl= parsed_parms['head']['transl'].to(args.device),
      head_scale_params=  torch.ones((n_frames,1)).to(args.device),
      device=args.device,
      img_size=(args.w,args.h),
      Ps=Ps_batch,
      return2d=True,
          )
    # denormalize the head keypoints 
    head_keys_proj=util.denormalize_keys_batch(keys=head_keys_proj,w=args.w,h=args.h)
    head_keys_proj=head_keys_proj.squeeze().cpu().detach().numpy()
    head_vs=head_vs.squeeze().cpu().detach().numpy()
    
    
    
    ### initialize and run MANO model ###
    rh_model = mano.model.load(
              model_path=mano_model_path,
              is_right= True, 
              num_pca_comps=args.n_pca, 
              batch_size=n_frames, 
              flat_hand_mean=True).to(args.device)
    
    [rh_vs, rh_keys_3ds, rh_keys_proj] = tracking_util.mano_forwarding(
              h_model=rh_model,
              betas=parsed_parms['right_hand']['betas'].to(args.device),
              transl= parsed_parms['right_hand']['transl'].to(args.device),
              rot= parsed_parms['right_hand']['global_orient'].to(args.device),
              pose= parsed_parms['right_hand']['hand_pose'].to(args.device),
              device=args.device,
              Ps=Ps_batch,  
              img_size=(args.w,args.h),
              return_2d=True,
          )

    rh_faces = rh_model.faces
    rh_keys_proj=util.denormalize_keys_batch(keys=rh_keys_proj,w=args.w,h=args.h)
    rh_keys_proj=rh_keys_proj.squeeze().cpu().detach().numpy()
    rh_vs=rh_vs.squeeze().cpu().detach().numpy()
    
    ### setup open3d camera ###
    vis, cam = o3du.view_setup_known_cam(ext=np.eye(4))
    
  for i in range(0,n_frames,args.show_every):
 
     
    img,seg_img= util.get_image(vid,seg,i)
     
    img = util.keypoint_overlay(
      np.concatenate((head_keys[i],rh_keys[i]),axis=0).astype(int),
      img)
    if args.vis3d:
      # draw keypoints on the image
      img = util.keypoint_overlay(
        head_keys_proj[i].astype(int),
        c=(255,0,0),
        img=img)
      img = util.keypoint_overlay(
        rh_keys_proj[i].astype(int),
        c=(255,0,0),
        img=img)
    if args.vis_mask:
      img = np.concatenate((img,seg_img),axis=1)  
    ### visualize 2D ###
    cv2.imshow('image',img)
    cv2.waitKey(1)
    if args.vis3d: 
      deformations = clean_deform_batch( deforms=deformations)
      deformations_cam =  tranu.convert_canonical_deforms_into_cam_space(
                      can_deforms =system_util.tor2np(deformations[i]),
                      head_pose =system_util.tor2np(head_poses[i]),
                      cam_transform = RT_np)
      
      head_vs_in_cam = tranu.apply_transform_np(head_vs[i],RT=RT_np)
      rh_vs_in_cam = tranu.apply_transform_np(rh_vs[i],RT=RT_np)  
      deformed_head_vs_in_cam = head_vs_in_cam-deformations_cam
      
      head_mesh = o3du.get_mesh_from_vs_and_faces(deformed_head_vs_in_cam,
                                                  flame_faces)
      rh_mesh = o3du.get_mesh_from_vs_and_faces(rh_vs_in_cam,rh_faces)
      if args.vis_con:
        head_mesh = o3du.get_mesh_colored_by_idx(
                  t_idx=head_con_idx[i],
                  t_c=[1, 0, 0],
                  mesh=head_mesh) 
        rh_mesh = o3du.get_mesh_colored_by_idx(
                  t_idx=rh_con_idx[i],
                  t_c=[1, 1, 0],
                  mesh=rh_mesh)
      
      # o3du.static_visualization(meshes=[head_mesh,rh_mesh],
      #                           frame=True,
      #                           frame_size=0.3)
    
      ### visualize 3D ###
      o3du.visualization_process_o3d(
          vis, [head_mesh,rh_mesh], cam) 
          
  return
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='configuratoins') 
    parser.add_argument('--dataset_path', type=str, default="../DecafDataset/") 
    parser.add_argument('--sub', type=str, default="S3")# subject name
    parser.add_argument('--mode', type=str, default="test")# train/test
    parser.add_argument('--vis2d', type=int, default=1)# visualize 2D
    parser.add_argument('--vis3d', type=int, default=1)# visualize 3D
    parser.add_argument('--vis_con', type=int, default=0)# visualize contacts
    parser.add_argument('--vis_mask', type=int, default=0)# visualize segmentation mask
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--n_pca', type=int, default=45)
    parser.add_argument('--w', type=int, default=1920)
    parser.add_argument('--h', type=int, default=1080)
    parser.add_argument('--cam', type=str, default="102")#camera id
    parser.add_argument('--show_every', type=int, default=5)#show every n frames
    args = parser.parse_args()
    args.dataset_path = "/mnt/d/DecafDataset"
    ### path setup ###
    asset_path = args.dataset_path+"/assets/"
    mano_model_path = asset_path+'/mano_v1_2/models/MANO_RIGHT.pkl'
    flame_model_path = asset_path+"/generic_model.pkl"
    flame_landmark_path = asset_path+"/landmark_embedding.npy"
     
    ignore_idx = np.load(asset_path+"/FLAME_neck_idx.npy")
    main(args,mode=args.mode,sub=args.sub)

 