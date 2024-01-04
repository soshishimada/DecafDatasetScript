


import system_util,argparse
import numpy as np

def get_consecutive_ids(cut_ids:list,n_frames:int) -> list:
  """_summary_

  Args:
      cut_ids (list):a list that indicates the start and end frame ids of each cut
      n_frames (int): a total number of frames in the video

  Returns:
      consecutive_ids (list): a list of lists that contains consecutive frame ids
  """
  consecutive_ids = []
   
  for i  in range(len(cut_ids)):
    if i==0:
      consecutive_ids.append(list(np.arange(0,cut_ids[i][0]+1)))
       
    else:
      consecutive_ids.append(list(np.arange(cut_ids[i-1][1],cut_ids[i][0]+1)))
       
  consecutive_ids.append(list(np.arange(cut_ids[-1][1],n_frames)))
  return consecutive_ids

def main(args):
  params=system_util.pickle_loader(args.dataset_path + '/'+args.mode+'/params/'+args.sub+'/params_smoothed.pkl')
  cut_ids=system_util.pickle_loader(args.dataset_path + '/'+args.mode+'/cutIDs/'+args.sub+'.pkl')  
  n_frames = len(params.keys())
  consecutive_ids = get_consecutive_ids(cut_ids,n_frames)
  
  for i in range(len(consecutive_ids)):
    print(consecutive_ids[i])

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
     
    main(args)