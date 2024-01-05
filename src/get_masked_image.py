 
import cv2,argparse

def main(vid_path:str,seg_cap:str)->None:    
    # load masking image and RGB image
    print("loading videos...")
    cap = cv2.VideoCapture(vid_path)  
    seg_cap = cv2.VideoCapture(seg_path) 
    res, image = cap.read()
    res, mask = seg_cap.read()
    if not res:
        raise Exception("Video not found")
    print("videos loaded.")
    
    # mask the image
    ret, mask = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)
    mask[mask!=0]=1  
    original_shape = image
    masked_image=image*mask
     
    cap.release() 
    seg_cap.release()
    # save image
    print("saving image..."	) 
    cv2.imwrite(args.save_path+'/rgb_image.jpg',original_shape)
    cv2.imwrite(args.save_path+'/masked_image.jpg',masked_image)
    print("image saved at "+args.save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='configuratoins') 
    parser.add_argument('--dataset_path', type=str, default="../DecafDataset/")  
    parser.add_argument('--save_path', type=str, default="./results/mask/")  
    args = parser.parse_args()
    # set path to the RGB vieo and segmentation video in DecafDataset
    vid_path = args.dataset_path+"/test/videos/S1/084.mp4"
    seg_path = args.dataset_path+"/test/segmentations/S1/084.mp4"  
     
    main(vid_path,seg_path) 