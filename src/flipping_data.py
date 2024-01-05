import os,argparse,torch,cv2,mano 
import numpy as np  
import FLAME_util,util,system_util,tracking_util
import open3d_util as o3du
import transformation_util as tranu 
import copy

def clean_deform_batch( deforms):
  """
  Input: deforms (*,5023,3)
  output: valid deforms (*,5023,3)
  """
  deforms[:, ignore_idx] = 0
  return deforms

def main(args,mode,sub):
  
  ###########################################
  ##### data loading and initialization #####
  ###########################################
  ### video loading ###
  vid_path = args.dataset_path + '/'+mode+'/videos/'+sub+'/'
  
  vid_names = os.listdir(vid_path)
  vid_names.sort()
   
  vid_name = [x for x in vid_names if args.cam == x[:-4]][0]
  vid = cv2.VideoCapture(vid_path+vid_name)
  n_frames=int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
 
  vid.set(cv2.CAP_PROP_POS_FRAMES, args.frame_id)
  ret, img = vid.read()
  
  ### contacts, deformation, parameters, camera loading ###
  contacts_head = np.load(args.dataset_path + '/'+mode+'/contacts/'+sub+'/contacts_head.npy')
  contacts_rh = np.load(args.dataset_path + '/'+mode+'/contacts/'+sub+'/contacts_rh.npy')
  contacts_lh = np.load(args.dataset_path + '/'+mode+'/contacts/'+sub+'/contacts_lh.npy')
  contacts_head_flipped = np.load(args.dataset_path + '/'+mode+'/contacts/'+sub+'/contacts_head_flipped.npy')   
  deformations = np.load(args.dataset_path + '/'+mode+'/deformations/'+sub+'/deforms.npy')
  params=system_util.pickle_loader(args.dataset_path + '/'+mode+'/params/'+sub+'/params_smoothed.pkl')
  cam_params= system_util.pickle_loader(args.dataset_path + '/'+mode+'/cameras/'+sub+'/cameras.pkl')
  Ps= np.matmul(cam_params[args.cam]['intrinsic'],
                cam_params[args.cam]['extrinsic'][:-1])
  Ps= torch.FloatTensor(Ps).to(args.device)
  Ps_batch = Ps.clone().view(1,  1, 3, 4).expand(n_frames, -1, -1, -1) 
  RT_np=np.array(cam_params[args.cam]['extrinsic']) 
  Ps_cam = np.matmul(cam_params[args.cam]['intrinsic'],
                np.eye(4)[:3])
  Ps_cam = torch.FloatTensor(Ps_cam).to(args.device)
  Ps_cam_batch = Ps_cam.clone().view(1,  1, 3, 4) 
  
  parsed_parms = util.params_parser(params)
  head_rot_mat = tranu.rot_angle2rotmat(parsed_parms['head']['global_orient']) 
  head_poses = tranu.convert_R_T_to_RT4x4(head_rot_mat,
                                          parsed_parms['head']['transl'])
  rh_con_idx =\
          [np.nonzero(contacts_rh[i])[0] for i in range(len(contacts_rh))]
  lh_con_idx =\
          [np.nonzero(contacts_lh[i])[0] for i in range(len(contacts_lh))]
  head_con_idx = [np.nonzero(contacts_head[i])[0]
                  for i in range(len(contacts_head))]
  head_con_idx_f= [np.nonzero(contacts_head_flipped[i])[0]
                  for i in range(len(contacts_head_flipped))]
  assert n_frames == \
        contacts_head.shape[0] == \
        contacts_rh.shape[0] == \
        contacts_lh.shape[0] == \
        contacts_head_flipped.shape[0] == \
        deformations.shape[0] == \
        len(params.keys())
    
  ### initialize and run FLAME model ###
  flame_model = FLAME_util.FLAME( 
                        flame_model_path,
                        flame_landmark_path,
                        ).to(args.device)
  flame_faces = FLAME_util.get_FLAME_faces(flame_model_path)
     

  [head_vs, _] = \
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
            return2d=False,
        )
 
   
  ### initialize and run MANO model ###
  rh_model = mano.model.load(
            model_path=rh_mano_model_path,
            is_right= True, 
            num_pca_comps=args.n_pca, 
            batch_size=n_frames, 
            flat_hand_mean=True).to(args.device)
  rh_faces = rh_model.faces 
  lh_model = mano.model.load(
            model_path=lh_mano_model_path,
            is_right= True, 
            num_pca_comps=args.n_pca, 
            batch_size=n_frames, 
            flat_hand_mean=True).to(args.device)
  lh_faces=lh_model.faces
  [rh_vs, rh_keys_3ds ] = tracking_util.mano_forwarding(
            h_model=rh_model,
            betas=parsed_parms['right_hand']['betas'].to(args.device),
            transl= parsed_parms['right_hand']['transl'].to(args.device),
            rot= parsed_parms['right_hand']['global_orient'].to(args.device),
            pose= parsed_parms['right_hand']['hand_pose'].to(args.device),
            device=args.device,
            Ps=Ps_batch,  
            img_size=(args.w,args.h),
            return_2d=False,
        )

 
   
  ###########################################
  ################ data flipping ############
  ###########################################
  
  #### flip the RGB image ####
  img_f=tranu.image_flipper_with_image_center(
    k=np.array(cam_params[args.cam]['intrinsic'])[:3,:3],
    img=img)
 
  ### transform the deformation into the camera frame -> apply flipping ###
  deformations = clean_deform_batch( deforms=deformations)
  deformations_cam =  tranu.convert_canonical_deforms_into_cam_space(
                  can_deforms =system_util.tor2np(deformations[args.frame_id]),
                  head_pose =system_util.tor2np(head_poses[args.frame_id]),
                  cam_transform = RT_np)
  
  deformation_cam_f = copy.copy(deformations_cam)
  deformation_cam_f[:, 0] *= -1
  deformation_cam_f = deformation_cam_f[lr_head_corresp_idx]
  
  ###  apply flipping on the head and hand vertices in camera frame ###
  head_vs_in_cam = tranu.apply_transform_np(head_vs[args.frame_id],RT=RT_np)
  rh_vs_in_cam = tranu.apply_transform_np(rh_vs[args.frame_id],RT=RT_np)  
  head_vs_in_cam_f = copy.copy(head_vs_in_cam)
  head_vs_in_cam_f[:, 0] *= -1
  head_vs_in_cam_f = head_vs_in_cam_f[lr_head_corresp_idx]
  lh_vs_in_cam = copy.copy(rh_vs_in_cam)
  lh_vs_in_cam[:, 0] *= -1
  
  
  ###  apply deformation on the face ###
  deformed_head_vs_in_cam = head_vs_in_cam-deformations_cam
  deformed_head_vs_in_cam_f = head_vs_in_cam_f-deformation_cam_f
  
 
  ### project the original and flipped vertices onto the image ###
  deformed_head_vs_in_cam_2d = tracking_util.multiview_projection_batch(
    Ps_cam_batch, system_util.np2tor(deformed_head_vs_in_cam)[None], 
    device=head_vs.device
    ).squeeze().cpu().detach().numpy()
  rh_vs_in_cam_2d = tracking_util.multiview_projection_batch(
    Ps_cam_batch, system_util.np2tor(rh_vs_in_cam)[None], 
    device=head_vs.device
    ).squeeze().cpu().detach().numpy()
  deformed_head_vs_in_cam_2d_f = tracking_util.multiview_projection_batch(
    Ps_cam_batch, system_util.np2tor(deformed_head_vs_in_cam_f)[None], 
    device=head_vs.device
    ).squeeze().cpu().detach().numpy()
  lh_vs_in_cam_2d = tracking_util.multiview_projection_batch(
    Ps_cam_batch, system_util.np2tor(lh_vs_in_cam)[None], 
    device=head_vs.device
    ).squeeze().cpu().detach().numpy()  
  
  ### draw the projected vertices ###
  img = util.keypoint_overlay(
    np.concatenate(
      (deformed_head_vs_in_cam_2d ,rh_vs_in_cam_2d ),axis=0
      ).astype(int),
      img,
      c=(128,128,128))
  img_f = util.keypoint_overlay(
    np.concatenate(
      (deformed_head_vs_in_cam_2d_f ,lh_vs_in_cam_2d ),axis=0
      ).astype(int),
      img_f)

  ### obtain meshes in Open3D format ###
  head_mesh = o3du.get_mesh_from_vs_and_faces(deformed_head_vs_in_cam,
                                              flame_faces)
  head_mesh_f = o3du.get_mesh_from_vs_and_faces(deformed_head_vs_in_cam_f,
                                                flame_faces,
                                                colors=[0,0.7,0])
  rh_mesh = o3du.get_mesh_from_vs_and_faces(rh_vs_in_cam,rh_faces)
  lh_mesh = o3du.get_mesh_from_vs_and_faces(lh_vs_in_cam,lh_faces,colors=[0,0.7,0])
  
  ### colorize the contacts ###
  head_mesh = o3du.get_mesh_colored_by_idx(
            t_idx=head_con_idx[args.frame_id],
            t_c=[1, 0, 0],
            mesh=head_mesh) 
  rh_mesh = o3du.get_mesh_colored_by_idx(
            t_idx=rh_con_idx[args.frame_id],
            t_c=[1, 1, 0],
            mesh=rh_mesh)
  head_mesh_f = o3du.get_mesh_colored_by_idx(
            t_idx=head_con_idx_f[args.frame_id],
            t_c=[1, 0, 0],
            mesh=head_mesh_f,
            keep_ori_color=True)
  lh_mesh = o3du.get_mesh_colored_by_idx(
            t_idx=lh_con_idx[args.frame_id],
            t_c=[1, 1, 0],
            mesh=lh_mesh,
            keep_ori_color=True)
  
  ### save image and meshes ###
  print("saving image and meshes at ",args.save_path)
  cv2.imwrite(args.save_path+"img.png",img)
  cv2.imwrite(args.save_path+"img_f.png",img_f)
  o3du.save_mesh_to_ply_o3d(args.save_path+"head_mesh.ply",head_mesh)
  o3du.save_mesh_to_ply_o3d(args.save_path+"rh_mesh.ply",rh_mesh)
  o3du.save_mesh_to_ply_o3d(args.save_path+"head_mesh_f.ply",head_mesh_f)
  o3du.save_mesh_to_ply_o3d(args.save_path+"lh_mesh.ply",lh_mesh)
  print("done")
  
  if args.vis_3d:
    o3du.static_visualization(meshes=[head_mesh,rh_mesh,lh_mesh,head_mesh_f],
                            frame=True,
                            frame_size=0.3)
  
  return
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='configuratoins') 
    parser.add_argument('--dataset_path', type=str, default="../DecafDataset/") 
    parser.add_argument('--save_path', type=str, default="./results/flip/") 
    parser.add_argument('--sub', type=str, default="S3")# subject name
    parser.add_argument('--mode', type=str, default="test")# train or test
    parser.add_argument('--device', type=str, default="cpu")# cuda or cpu
    parser.add_argument('--vis_3d', type=int, default=1)# visualize 3d or not
    parser.add_argument('--n_pca', type=int, default=45)
    parser.add_argument('--w', type=int, default=1920)# image width
    parser.add_argument('--h', type=int, default=1080)# image height
    parser.add_argument('--cam', type=str, default="102")#camera id
    parser.add_argument('--frame_id', type=int, default=90)#target frame id
    args = parser.parse_args() 
    ### path setup ###
    asset_path = args.dataset_path+"/assets/"
    rh_mano_model_path = asset_path+'/mano_v1_2/models/MANO_RIGHT.pkl'
    lh_mano_model_path = asset_path+'/mano_v1_2/models/MANO_LEFT.pkl'
    flame_model_path = asset_path+"/generic_model.pkl"
    flame_landmark_path = asset_path+"/landmark_embedding.npy" 
    ignore_idx = np.load(asset_path+"/FLAME_neck_idx.npy")
    lr_head_corresp_idx = np.load(asset_path+'left_right_face_corresps.npy')
     
    main(args,mode=args.mode,sub=args.sub)

 