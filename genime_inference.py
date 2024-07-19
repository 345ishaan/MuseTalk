import argparse
import os
from omegaconf import OmegaConf
import numpy as np
import cv2
import uuid
import torch
import glob
import pickle
from tqdm import tqdm
import subprocess
import copy
import requests
import wave
import pdb

from musetalk.utils.utils import get_file_type,get_video_fps,datagen
from musetalk.utils.preprocessing import get_landmark_and_bbox,read_imgs,coord_placeholder
from musetalk.utils.blending import get_image
from musetalk.utils.utils import load_all_model
import shutil

# load model weights

@torch.no_grad()
def infer(image_urls, audio_urls, save_dir, batch_size=8, fps=25, bbox_shift=0, **kwargs):
    audio_processor, vae, unet, positional_embedding = load_all_model()
    positional_embedding = positional_embedding.half()
    vae.vae = vae.vae.half()
    unet.model = unet.model.half()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    video_fnames = []
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for iter, (image_url, audio_url) in enumerate(zip(image_urls, audio_urls)):
        response = requests.get(image_url)
        if response.status_code == 200:
            image_data = np.asarray(bytearray(response.content), dtype="uint8")
            image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            assert image is not None
            input_img_list = [image]
        else:
            raise Exception(f"Failed to download image blob {image_url}")
        ############################################## extract audio feature ##############################################
        audio_res = requests.get(audio_url)
        if audio_res.status_code == 200:
           audio_path = f"/tmp/{str(uuid.uuid4())}.wav"
           fp = open(audio_path, "wb")
           fp.write(audio_res.content)
           fp.close()
        else:
            raise Exception(f"Can't parse audio blob url {audio_url}")
        whisper_feature = audio_processor.audio2feat(audio_path)
        whisper_chunks = audio_processor.feature2chunks(feature_array=whisper_feature,fps=fps)
        ############################################## preprocess input image  ##############################################
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift, read=False)
        i = 0
        input_latent_list = []
        for bbox, frame in zip(coord_list, frame_list):
            if bbox == coord_placeholder:
                continue
            x1, y1, x2, y2 = bbox
            crop_frame = frame[y1:y2, x1:x2]
            crop_frame = cv2.resize(crop_frame,(256,256),interpolation = cv2.INTER_LANCZOS4)
            latents = vae.get_latents_for_unet(crop_frame)
            input_latent_list.append(latents)
    
        # to smooth the first and the last frame
        frame_list_cycle = frame_list + frame_list[::-1]
        coord_list_cycle = coord_list + coord_list[::-1]
        input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        ############################################## inference batch by batch ##############################################
        print("start inference")
        video_num = len(whisper_chunks)
        gen = datagen(whisper_chunks,input_latent_list_cycle,batch_size)
        res_frame_list = []
        timesteps = torch.tensor([0], device=device)
        for i, (whisper_batch,latent_batch) in enumerate(tqdm(gen,total=int(np.ceil(float(video_num)/batch_size)))):
            audio_feature_batch = torch.from_numpy(whisper_batch)
            audio_feature_batch = audio_feature_batch.to(device=unet.device,
                                                         dtype=unet.model.dtype) # torch, B, 5*N,384
            audio_feature_batch = positional_embedding(audio_feature_batch)
            latent_batch = latent_batch.to(dtype=unet.model.dtype)
            
            pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
            recon = vae.decode_latents(pred_latents)
            for res_frame in recon:
                res_frame_list.append(res_frame)
                
        ############################################## pad to full image ##############################################
        print("pad talking image to original video")
        # make a temp directory to save images.
        result_img_save_path = f"/tmp/{str(uuid.uuid4())}"
        os.makedirs(result_img_save_path)

        for i, res_frame in enumerate(tqdm(res_frame_list)):
            bbox = coord_list_cycle[i%(len(coord_list_cycle))]
            ori_frame = copy.deepcopy(frame_list_cycle[i%(len(frame_list_cycle))])
            x1, y1, x2, y2 = bbox
            try:
                res_frame = cv2.resize(res_frame.astype(np.uint8),(x2-x1,y2-y1))
            except:
#                 print(bbox)
                continue
            
            combine_frame = get_image(ori_frame,res_frame,bbox)
            cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png",combine_frame)

        cmd_img2video = f"/MuseTalk/ffmpeg-7.0.1-amd64-static/ffmpeg -y -v warning -r {fps} -f image2 -i {result_img_save_path}/%08d.png -vcodec libx264 -vf format=rgb24,scale=out_color_matrix=bt709,format=yuv420p -crf 18 temp.mp4"
        print(cmd_img2video)
        os.system(cmd_img2video)
        

        # make a temporary file name to save combined audio and video for cur instance.
        output_vid_name = os.path.join(save_dir, f"{iter}.mp4")
        cmd_combine_audio = f"/MuseTalk/ffmpeg-7.0.1-amd64-static/ffmpeg -y -v warning -i {audio_path} -i temp.mp4 {output_vid_name}"
        print(cmd_combine_audio)
        os.system(cmd_combine_audio)
        
        os.remove("temp.mp4")
        os.remove(audio_path)
        shutil.rmtree(result_img_save_path)
        video_fnames.append(output_vid_name)


    concat_file = os.path.join(save_dir, 'concat_list.txt')
    final_concat_fname = os.path.join(save_dir, 'final_concat.mp4')
    with open(concat_file, 'w') as f:
        for fname in video_fnames:
            f.write(f"file '{fname}'\n")
    concat_cmd = f"/MuseTalk/ffmpeg-7.0.1-amd64-static/ffmpeg -y -safe 0 -f concat -i {concat_file} -c copy {final_concat_fname}"
    os.system(concat_cmd)



def run_infer(image_urls, audio_urls, save_fname):
    infer(image_urls, audio_urls, save_fname)


if __name__ == "__main__":
    image_urls = [
        "https://ttvaarlnqssopdguetwq.supabase.co/storage/v1/object/sign/genime-bucket/character_watson.webp?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1cmwiOiJnZW5pbWUtYnVja2V0L2NoYXJhY3Rlcl93YXRzb24ud2VicCIsImlhdCI6MTcyMTIwMTg0NiwiZXhwIjoxNzUyNzM3ODQ2fQ._ylQ-43B20ctguDbDcYKa7CDRq2Dje0nuquFl11qFNM&t=2024-07-17T07%3A37%3A26.919Z",
        "https://ttvaarlnqssopdguetwq.supabase.co/storage/v1/object/sign/genime-bucket/character_sherlock.webp?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1cmwiOiJnZW5pbWUtYnVja2V0L2NoYXJhY3Rlcl9zaGVybG9jay53ZWJwIiwiaWF0IjoxNzIwODgyODM1LCJleHAiOjE3NTI0MTg4MzV9.SuT_J2Btf5pKoExbxd-iu8KhA7Q3_JirVEbZhnv79m0&t=2024-07-13T15%3A00%3A36.083Z"
    ]
    image_urls = [
        "https://ttvaarlnqssopdguetwq.supabase.co/storage/v1/object/sign/genime-bucket/character_tushar.webp?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1cmwiOiJnZW5pbWUtYnVja2V0L2NoYXJhY3Rlcl90dXNoYXIud2VicCIsImlhdCI6MTcyMTI5NDQzMiwiZXhwIjoxNzUyODMwNDMyfQ.S12pYdgP89LSr65YAv7rfY2_xsCbdaa7wWG3aTxpMS0&t=2024-07-18T09%3A20%3A32.159Z",
        "https://ttvaarlnqssopdguetwq.supabase.co/storage/v1/object/sign/genime-bucket/character_ishan2.webp?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1cmwiOiJnZW5pbWUtYnVja2V0L2NoYXJhY3Rlcl9pc2hhbjIud2VicCIsImlhdCI6MTcyMTM3NjkzMywiZXhwIjoxNzUyOTEyOTMzfQ.qHguLkQe09QFdYd9ZMQ-mD9m8UYOZ5FFJD_1PZlTsFM&t=2024-07-19T08%3A15%3A33.669Z"
    ]
    audio_urls = [
        "https://ttvaarlnqssopdguetwq.supabase.co/storage/v1/object/sign/genime-bucket/15.wav?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1cmwiOiJnZW5pbWUtYnVja2V0LzE1LndhdiIsImlhdCI6MTcyMTIwMTMzNiwiZXhwIjoxNzUyNzM3MzM2fQ.bVOALJpkftbX5fAkkVTbLLjbaMVBwWmQmfhTFO1o65I",
        "https://ttvaarlnqssopdguetwq.supabase.co/storage/v1/object/sign/genime-bucket/18.wav?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1cmwiOiJnZW5pbWUtYnVja2V0LzE4LndhdiIsImlhdCI6MTcyMTIwMTg3NiwiZXhwIjoxNzUyNzM3ODc2fQ.GITPWaOL-xn24unxvWkJX9XNE_7XzHRqoja8Zd-Wnq4"
    ]
    image_urls_2 = [
        "https://ttvaarlnqssopdguetwq.supabase.co/storage/v1/object/sign/genime-bucket/character_watson.webp?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1cmwiOiJnZW5pbWUtYnVja2V0L2NoYXJhY3Rlcl93YXRzb24ud2VicCIsImlhdCI6MTcyMTIwMTg0NiwiZXhwIjoxNzUyNzM3ODQ2fQ._ylQ-43B20ctguDbDcYKa7CDRq2Dje0nuquFl11qFNM&t=2024-07-17T07%3A37%3A26.919Z",
        "https://ttvaarlnqssopdguetwq.supabase.co/storage/v1/object/sign/genime-bucket/character_sherlock.webp?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1cmwiOiJnZW5pbWUtYnVja2V0L2NoYXJhY3Rlcl9zaGVybG9jay53ZWJwIiwiaWF0IjoxNzIwODgyODM1LCJleHAiOjE3NTI0MTg4MzV9.SuT_J2Btf5pKoExbxd-iu8KhA7Q3_JirVEbZhnv79m0&t=2024-07-13T15%3A00%3A36.083Z",
        "https://ttvaarlnqssopdguetwq.supabase.co/storage/v1/object/sign/genime-bucket/character_watson.webp?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1cmwiOiJnZW5pbWUtYnVja2V0L2NoYXJhY3Rlcl93YXRzb24ud2VicCIsImlhdCI6MTcyMTIwMTg0NiwiZXhwIjoxNzUyNzM3ODQ2fQ._ylQ-43B20ctguDbDcYKa7CDRq2Dje0nuquFl11qFNM&t=2024-07-17T07%3A37%3A26.919Z",
        "https://ttvaarlnqssopdguetwq.supabase.co/storage/v1/object/sign/genime-bucket/character_sherlock.webp?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1cmwiOiJnZW5pbWUtYnVja2V0L2NoYXJhY3Rlcl9zaGVybG9jay53ZWJwIiwiaWF0IjoxNzIwODgyODM1LCJleHAiOjE3NTI0MTg4MzV9.SuT_J2Btf5pKoExbxd-iu8KhA7Q3_JirVEbZhnv79m0&t=2024-07-13T15%3A00%3A36.083Z"
    ]
    audio_urls_2 = [
        "https://ttvaarlnqssopdguetwq.supabase.co/storage/v1/object/sign/genime-bucket/39.wav?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1cmwiOiJnZW5pbWUtYnVja2V0LzM5LndhdiIsImlhdCI6MTcyMTIwNjg5NCwiZXhwIjoxNzUyNzQyODk0fQ.cVn4bzAwD4LxHFdW_FNopz4B_JlxwjffyJi2kboyAak",
        "https://ttvaarlnqssopdguetwq.supabase.co/storage/v1/object/sign/genime-bucket/38.wav?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1cmwiOiJnZW5pbWUtYnVja2V0LzM4LndhdiIsImlhdCI6MTcyMTIwNjg4OSwiZXhwIjoxNzUyNzQyODg5fQ.b3ZoUItz3sckW371HDhUL1bOnRA5GqNkTSWfeml931A",
        "https://ttvaarlnqssopdguetwq.supabase.co/storage/v1/object/sign/genime-bucket/37.wav?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1cmwiOiJnZW5pbWUtYnVja2V0LzM3LndhdiIsImlhdCI6MTcyMTIwNjg4MiwiZXhwIjoxNzUyNzQyODgyfQ.fDvsUZiqcedek-z4_j4QQgtRwbJ0k2c8oc7k8gCiQPQ",
        "https://ttvaarlnqssopdguetwq.supabase.co/storage/v1/object/sign/genime-bucket/36.wav?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1cmwiOiJnZW5pbWUtYnVja2V0LzM2LndhdiIsImlhdCI6MTcyMTIwNjg3NywiZXhwIjoxNzUyNzQyODc3fQ.K8tPi8RFQ7_66BjfJVfR9atq-oVIKW3xBhHH-qjEcb8"
    ]
    save_dir = "/MuseTalk/results_w_fix"
    run_infer(image_urls, audio_urls, save_dir)



