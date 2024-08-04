import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os

def sam2_fn(video_path, frames_dir, pts=[[500, 180], [580, 150]]):
    video_name = video_path.split('/')[-1].split('.')[0]
    frames_dir = os.path.join(frames_dir, video_name)

    # use bfloat16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    from sam2.build_sam import build_sam2_video_predictor

    sam2_checkpoint = "../checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

    def show_mask(mask, ax, obj_id=None, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            cmap = plt.get_cmap("tab10")
            cmap_idx = 0 if obj_id is None else obj_id
            color = np.array([*cmap(cmap_idx)[:3], 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)


    def show_points(coords, labels, ax, marker_size=200):
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

    print('Converting video to frames...')
    raw_images_dir = os.path.join(frames_dir, 'raw_images')
    os.makedirs(raw_images_dir, exist_ok=True)
    os.system(f"ffmpeg -i {video_path} -q:v 2 -start_number 0 {raw_images_dir}/'%05d.jpg'")

    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(raw_images_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    print("Performing Inference...")
    inference_state = predictor.init_state(video_path=raw_images_dir)

    predictor.reset_state(inference_state)

    ann_frame_idx = 0  # the frame index we interact with
    ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

    # Let's add a 2nd positive click at (x, y) = (250, 220) to refine the mask
    # sending all clicks (and their labels) to `add_new_points`
    points = np.array(pts, dtype=np.float32)
    # for labels, `1` means positive click and `0` means negative click
    labels = np.array([1]*len(pts), np.int32)
    _, out_obj_ids, out_mask_logits = predictor.add_new_points(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )

    print("Propagating video....")
    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    vis_frame_stride = 5 

    def format_mask(mask, obj_id=None,):
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w)# * color.reshape(1, 1, -1)
        return mask_image

    print("Saving final segmented frames for LLM....")
    final_segmented_video_dir = os.path.join(frames_dir, 'final_segmented_video')
    os.makedirs(final_segmented_video_dir, exist_ok=True)
    output_file = os.path.join(final_segmented_video_dir, 'final_segmented_video.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30  # Frames per second
    frame_size = (360, 640)  # Width and height of the frames
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, frame_size)

    for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
        org_img = cv2.imread(os.path.join(raw_images_dir, frame_names[out_frame_idx]))
        final_mask = np.zeros((org_img.shape[:-1]), dtype = bool)
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            out_mask = format_mask(out_mask)
            final_mask = final_mask | out_mask
        # print("Org image shape:", org_img.shape)
        # print("Final mask shape: ", final_mask.shape)
        final_image = org_img * final_mask[:,:,np.newaxis]
        # plt.imshow(final_image)
        cv2.imwrite(os.path.join(final_segmented_video_dir, f'{out_frame_idx}.jpg'), final_image)
        video_writer.write(final_image)
    video_writer.release()
    # print(f"Video saved as {output_file}")   

    def save_mask(img, mask, final_orange_video_dir, fnumber, obj_id=None, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            cmap = plt.get_cmap("tab10")
            cmap_idx = 0 if obj_id is None else obj_id
            color = np.array([*cmap(cmap_idx)[:3], 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        fig, ax = plt.subplots()
        plt.imshow(img)
        plt.imshow(mask_image)
        plt.axis('off')
        plt.savefig(os.path.join(final_orange_video_dir, f'{fnumber}.jpg'), bbox_inches='tight')

    print("Saving final segmented frames for visualization....")
    final_orange_video_dir = os.path.join(frames_dir, 'final_orange_video')
    os.makedirs(final_orange_video_dir, exist_ok=True)
    plt.close("all")
    for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
        plt.figure(figsize=(6, 4))
        # plt.title(f"frame {out_frame_idx}")
        orig_img = Image.open(os.path.join(raw_images_dir, frame_names[out_frame_idx]))
        final_mask = np.zeros((org_img.shape[:-1]), dtype = bool)
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            out_mask = format_mask(out_mask)
            final_mask = final_mask | out_mask
        save_mask(orig_img, final_mask, final_orange_video_dir, out_frame_idx, obj_id=out_obj_id)
    
    os.system(f"ffmpeg -framerate 10 -pattern_type glob -i '{final_segmented_video_dir}/*.jpg' -vf 'scale=iw/2*2:ih/2*2' -c:v libx264 -pix_fmt yuv420p {frames_dir}/final_video.mp4")
    os.system(f"ffmpeg -framerate 10 -pattern_type glob -i '{final_orange_video_dir}/*.jpg' -vf 'scale=iw/2*2:ih/2*2' -c:v libx264 -pix_fmt yuv420p {frames_dir}/final_orange_video.mp4")
if __name__ == '__main__':
    video_path = "./Flamingo.mp4"
    frames_dir = './frames_dir'
    pts = [[100, 140], [100, 120]]
    sam2_fn(video_path, frames_dir, pts)
