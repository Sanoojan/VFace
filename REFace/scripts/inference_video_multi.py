import argparse, os, sys
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
# import pandas as pd
from PIL import Image
from tqdm import tqdm, trange
# from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
import torchvision
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim_w_inv import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
import albumentations as A
import torchvision.transforms as transforms
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from moviepy.editor import AudioFileClip, VideoFileClip
import proglog
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor
from src.utils.alignmengt import crop_faces, calc_alignment_coefficients, crop_faces_from_image
from ldm.data.video_swap_dataset import VideoDataset
from torchvision.transforms import Resize
import torchvision.transforms.functional as TF 
from PIL import Image
from torchvision.transforms import PILToTensor
from pretrained.face_parsing.face_parsing_demo import init_faceParsing_pretrained_model, faceParsing_demo, vis_parsing_maps
import torch.nn as nn 
import yaml
from glob import glob
from scripts.face_swap_utils import *
from scripts.temporal_flow import *
import torchvision.transforms as transforms

# load safety model
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)


def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return torchvision.transforms.Compose(transform_list)

def crop_and_align_face(target_files):
    image_size = 1024
    scale = 1.0
    center_sigma = 0
    xy_sigma = 0
    use_fa = False
    
    print('Aligning images')
    crops, orig_images, quads = crop_faces(image_size, target_files, scale, center_sigma=center_sigma, xy_sigma=xy_sigma, use_fa=use_fa)
    
    inv_transforms = [
        calc_alignment_coefficients(quad + 0.5, [[0, 0], [0, image_size], [image_size, image_size], [image_size, 0]])
        for quad in quads
    ]
    
    return crops, orig_images, quads, inv_transforms

def crop_and_align_face_img(frame):
    image_size = 1024
    scale = 1.0
    center_sigma = 0
    xy_sigma = 0
    use_fa = False
    
    print('Aligning images')
    crops, orig_images, quads = crop_faces_from_image(image_size, frame, scale, center_sigma=center_sigma, xy_sigma=xy_sigma, use_fa=use_fa)
    
    inv_transforms = [
        calc_alignment_coefficients(quad + 0.5, [[0, 0], [0, image_size], [image_size, image_size], [image_size, 0]])
        for quad in quads
    ]
    
    return crops, orig_images, quads, inv_transforms

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept

def un_norm_clip(x1):
    x = x1*1.0 # to avoid changing the original tensor or clone() can be used
    reduce=False
    if len(x.shape)==3:
        x = x.unsqueeze(0)
        reduce=True
    x[:,0,:,:] = x[:,0,:,:] * 0.26862954 + 0.48145466
    x[:,1,:,:] = x[:,1,:,:] * 0.26130258 + 0.4578275
    x[:,2,:,:] = x[:,2,:,:] * 0.27577711 + 0.40821073
    
    if reduce:
        x = x.squeeze(0)
    return x

def un_norm(x):
    return (x+1.0)/2.0

def save_clip_img(img, path,clip=True):
    if clip:
        img=un_norm_clip(img)
    else:
        img=torch.clamp(un_norm(img), min=0.0, max=1.0)
    img = img.cpu().numpy().transpose((1, 2, 0))
    img = (img * 255).astype(np.uint8)
    img = Image.fromarray(img)
    img.save(path)

def run_inference(model, sampler, opt, device, config):
    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir
    Base_path=opt.Base_dir

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    # sample_path = os.path.join(outpath, "samples")
    result_path = os.path.join(outpath, "results")
    model_out_path = os.path.join(outpath, "model_outputs")
    target_video_name=os.path.basename(opt.target_video).split('.')[0]
    src_name=os.path.basename(opt.src_image).split('.')[0]
    
    target_frames_path=os.path.join(Base_path, target_video_name)
    target_cropped_face_path=os.path.join(Base_path, target_video_name+"cropped_face")
    mask_frames_path=os.path.join(Base_path, target_video_name+"mask_frames")
    # os.makedirs(sample_path, exist_ok=True)
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(target_frames_path, exist_ok=True)
    os.makedirs(mask_frames_path, exist_ok=True)
    os.makedirs(target_cropped_face_path, exist_ok=True)
    os.makedirs(model_out_path, exist_ok=True)
    out_video_filepath=os.path.join(outpath, target_video_name+'to'+src_name+'_swap.mp4')
    
    video_forcheck = VideoFileClip(opt.target_video)

    if video_forcheck.audio is None:
        no_audio = True
    else:
        no_audio = False

    del video_forcheck

    if not no_audio:
        video_audio_clip = AudioFileClip(opt.target_video)
    
    video = cv2.VideoCapture(opt.target_video)
    # video_shape
    video_shape = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    temp_results_dir = os.path.join(outpath, 'temp_results')
    inverse_results_dir=os.path.join(opt.Base_dir, 'inverse_results')
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    # fps = video.get(cv2.CAP_PROP_FPS)
    fps=10
    os.makedirs(temp_results_dir, exist_ok=True)
    os.makedirs(inverse_results_dir, exist_ok=True)
    
    faceParsing_model = init_faceParsing_pretrained_model(opt.faceParser_name, opt.faceParsing_ckpt, opt.segnext_config)
    
    
    crops, orig_images, quads, inv_transforms = crop_and_align_face([opt.src_image])
    crops = [crop.convert("RGB") for crop in crops]
    T = crops[0]
    src_image_new=os.path.join(temp_results_dir, src_name+'.png')
    T.save(src_image_new)

    
    # src_image=cv2.imread(opt.src_image)
    pil_im = Image.open(src_image_new).convert("RGB").resize((1024,1024), Image.BILINEAR)
    mask = faceParsing_demo(faceParsing_model, pil_im, convert_to_seg12=opt.seg12, model_name=opt.faceParser_name)
    Image.fromarray(mask).save(os.path.join(temp_results_dir, os.path.basename(opt.src_image)))

    
    # get count of mask_frames_path
    base_count = len(os.listdir(target_frames_path))
    mask_count= len(os.listdir(mask_frames_path))
    
    frame_count = opt.n_frames
    
    
    if base_count != frame_count or mask_count != frame_count :
        inv_transforms_all = []
        
        for frame_index in tqdm(range(frame_count)):
            ret, frame = video.read() 
            # if frame_index <1088:
            #     continue
            if ret:
                try:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_old = frame
                    Image.fromarray(frame).save(os.path.join(target_frames_path, f'{frame_index}.png'))
                    crops, orig_images, quads, inv_transforms = crop_and_align_face([os.path.join(target_frames_path, f'{frame_index}.png')])
                    
                    crops = [crop.convert("RGB") for crop in crops]
                    T = crops[0]
                    inv_transforms_all.append(inv_transforms[0])
                    
                    pil_im = T.resize((1024,1024), Image.BILINEAR)
                    mask = faceParsing_demo(faceParsing_model, pil_im, convert_to_seg12=opt.seg12, model_name=opt.faceParser_name)
                    Image.fromarray(mask).save(os.path.join(mask_frames_path, f'{frame_index}.png'))
                    # save T
                    T.save(os.path.join(target_cropped_face_path, f'{frame_index}.png'))
                except:
                    Image.fromarray(frame_old).save(os.path.join(target_frames_path, f'{frame_index}.png'))
                    inv_transforms_all.append(inv_transforms[0])
                    Image.fromarray(mask).save(os.path.join(mask_frames_path, f'{frame_index}.png'))
                    # save T
                    T.save(os.path.join(target_cropped_face_path, f'{frame_index}.png'))
                    print('error finding face at', frame_index)
                    pass
        # save inv_transforms_all
        np.save(os.path.join(Base_path,  target_video_name+'_inv_transforms.npy'), inv_transforms_all)
        
    # load inv_transforms_all
    inv_transforms_all=np.load(os.path.join(Base_path,  target_video_name+'_inv_transforms.npy'), allow_pickle=True)
    video.release()
    del faceParsing_model
    
    
    ################### Get reference
    conf_file=OmegaConf.load(opt.config)
    trans=A.Compose([
            A.Resize(height=224,width=224)])
    ref_img_path = src_image_new
    img_p_np=cv2.imread(ref_img_path)
    # ref_img = Image.open(ref_img_path).convert('RGB').resize((224,224))
    ref_img = cv2.cvtColor(img_p_np, cv2.COLOR_BGR2RGB)
    # ref_img= cv2.resize(ref_img, (224, 224))
    
    ref_mask_path = os.path.join(temp_results_dir, os.path.basename(opt.src_image))
    ref_mask_img = Image.open(ref_mask_path).convert('L')
    ref_mask_img = np.array(ref_mask_img)  # Convert the label to a NumPy array if it's not already

    # Create a mask to preserve values in the 'preserve' list
    # preserve = [1,2,4,5,8,9,17 ]
    preserve = conf_file.data.params.test.params['preserve_mask_src_FFHQ']
    # preserve = [1,2,4,5,8,9 ]
    ref_mask= np.isin(ref_mask_img, preserve)

    # Create a converted_mask where preserved values are set to 255
    ref_converted_mask = np.zeros_like(ref_mask_img)
    ref_converted_mask[ref_mask] = 255
    ref_converted_mask=Image.fromarray(ref_converted_mask).convert('L')
    # convert to PIL image
    
    #Gray Mask
    reference_mask_tensor=get_tensor(normalize=False, toTensor=True)(ref_converted_mask)
    mask_ref=transforms.Resize((224,224))(reference_mask_tensor)
    ref_img=trans(image=ref_img)
    ref_img=Image.fromarray(ref_img["image"])
    ref_img=get_tensor_clip()(ref_img)
    ref_img=ref_img*mask_ref
    ref_image_tensor = ref_img.to(device,non_blocking=True).to(torch.float16).unsqueeze(0)
    
    ######## Src Reconstruction ###########
    ref_img_inv_ori=Image.open(ref_img_path).convert("RGB").resize((512,512), Image.BILINEAR)
    ref_img_inv_ori = get_tensor()(ref_img_inv_ori)
    ref_img_inv_ori= ref_img_inv_ori*reference_mask_tensor
    
    ref_img_inv_ori_inpaint=ref_img_inv_ori.clone()
    ref_img_inv_ori_inpaint=ref_img_inv_ori_inpaint*(1-reference_mask_tensor)
    #######################################
    
    
    #Black_mask
    # ref_mask_img=Image.fromarray(ref_img).convert('L')
    # ref_mask_img_r = ref_converted_mask.resize(img_p_np.shape[1::-1], Image.NEAREST)
    # ref_mask_img_r = np.array(ref_mask_img_r)
    # ref_img[ref_mask_img_r==0]=0
    # ref_img=trans(image=ref_img)
    # ref_img=Image.fromarray(ref_img["image"])
    # ref_img=get_tensor_clip()(ref_img)
    # ref_image_tensor = ref_img.to(device,non_blocking=True).to(torch.float16).unsqueeze(0)
    ########################
    

    test_args=conf_file.data.params.test.params
    
    
    
    
    test_dataset=VideoDataset(data_path=target_cropped_face_path,mask_path=mask_frames_path,**test_args)
    test_dataloader= torch.utils.data.DataLoader(test_dataset, 
                                        batch_size=batch_size, 
                                        num_workers=4, 
                                        pin_memory=True, 
                                        shuffle=False,#sampler=train_sampler, 
                                        drop_last=True)





    start_code = None
    if opt.fixed_code:
        print("Using fixed code.......")
        start_code1= torch.randn([ opt.C, opt.H // opt.f, opt.W // opt.f], device=device)
        # extend the start code to batch size
        start_code1 = start_code1.unsqueeze(0).repeat(batch_size, 1, 1, 1)

   
    
    use_prior=False
    use_ddim_inversion=True
    
    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    sample=0
    
    
    # cond_only_once=False

    
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                all_samples = list()
  
  
                for batch_id, (test_batch,prior, test_model_kwargs,segment_id_batch) in enumerate(test_dataloader):
                    sample+=opt.n_samples
                    # if sample<980:
                    #     continue
                    # breakpoint()
                    
                        
                    test_model_kwargs={n:test_model_kwargs[n].to(device,non_blocking=True) for n in test_model_kwargs }
                    uc = None
                    if opt.scale != 1.0:
                        uc = model.learnable_vector.repeat(test_batch.shape[0],1,1)
                        if model.stack_feat:
                            uc2=model.other_learnable_vector.repeat(test_batch.shape[0],1,1)
                            uc=torch.cat([uc,uc2],dim=-1)
                    
                    # c = model.get_learned_conditioning(test_model_kwargs['ref_imgs'].squeeze(1).to(torch.float16))
                    landmarks=model.get_landmarks(test_batch) if model.Landmark_cond else None
                    ref_imgs=ref_image_tensor
                    # stack it ref_imgs to the shape of test_batch
                    ref_imgs=ref_imgs.repeat(test_batch.shape[0],1,1,1)
                    
                    ##############  Src Reconstruction ##########
                    ref_img_inv=ref_img_inv_ori
                    landmarks_src=model.get_landmarks(ref_img_inv.unsqueeze(0)) 
                    cond_w_src=model.conditioning_with_feat(ref_imgs[0].unsqueeze(0).to(torch.float32),landmarks=landmarks_src,tar=ref_img_inv.unsqueeze(0).to("cuda").to(torch.float32)).float()
                    cond_w_src=cond_w_src.repeat(test_batch.shape[0],1,1)
                    ############################
                    
                    
                    c=model.conditioning_with_feat(ref_imgs.squeeze(1).to(torch.float32),landmarks=landmarks,tar=test_batch.to("cuda").to(torch.float32)).float()
                    if (model.land_mark_id_seperate_layers or model.sep_head_att) and opt.scale != 1.0:
            
                        # concat c, landmarks
                        landmarks=landmarks.unsqueeze(1) if len(landmarks.shape)!=3 else landmarks
                        uc=torch.cat([uc,landmarks],dim=-1)
                    
                    
                    if c.shape[-1]==1024:
                        c = model.proj_out(c)
                    if len(c.shape)==2:
                        c = c.unsqueeze(1)
                    inpaint_image=test_model_kwargs['inpaint_image']
                    inpaint_mask=test_model_kwargs['inpaint_mask']
                    z_inpaint = model.encode_first_stage(test_model_kwargs['inpaint_image'])
                    z_inpaint = model.get_first_stage_encoding(z_inpaint).detach()
                    test_model_kwargs['inpaint_image']=z_inpaint
                    test_model_kwargs['inpaint_mask']=Resize([z_inpaint.shape[-1],z_inpaint.shape[-1]])(test_model_kwargs['inpaint_mask'])

                    ######## Src Reconstruction ###########
                    inpaint_mask_src=1-reference_mask_tensor
                    inpaint_image_src=ref_img_inv_ori_inpaint
                    inpaint_image_src=inpaint_image_src.unsqueeze(0).to(device)
                    z_inpaint_src = model.encode_first_stage(inpaint_image_src)
                    z_inpaint_src = model.get_first_stage_encoding(z_inpaint_src).detach()
                    inpaint_image_src=z_inpaint_src
                    inpaint_mask_src=Resize([z_inpaint_src.shape[-1],z_inpaint_src.shape[-1]])(inpaint_mask_src)
                    #######################################
                    
                    
                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                    inverse_cond=None
                    
                    if opt.Start_from_target:
                        print(" from target....")
                        x=test_batch
                        x=x.to(device)
                        encoder_posterior = model.encode_first_stage(x)
                        z = model.get_first_stage_encoding(encoder_posterior)
                        
                        
                        t=int(opt.target_start_noise_t)
                        # t = torch.ones((x.shape[0],), device=device).long()*t
                        t = torch.randint(t-1, t, (x.shape[0],), device=device).long()
                    
                        if use_ddim_inversion:
                            prior=prior.to(device)
                            encoder_posterior_2=model.encode_first_stage(prior)
                            z2 = model.get_first_stage_encoding(encoder_posterior_2)
                            test_batch_clip=test_batch
                            test_batch_clip=test_batch_clip.to(device)
                            test_batch_clip=test_batch_clip*(1-inpaint_mask)
                            test_batch_clip=un_norm(test_batch_clip)
                            test_batch_clip=Resize([224,224])(test_batch_clip)
                            test_batch_clip=TF.normalize(test_batch_clip, [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
                            
                            # visualize ref_imgs

                            inverse_cond=model.conditioning_with_feat(test_batch_clip.to(torch.float32),landmarks=landmarks,tar=test_batch.to("cuda").to(torch.float32)).float()
                            
                            inverse_steps=50
                            inverse_results_dir_for_batch=os.path.join(inverse_results_dir, str(batch_id))
                            prior=prior.to(device)
                            encoder_posterior_2=model.encode_first_stage(prior)
                            z2 = model.get_first_stage_encoding(encoder_posterior_2)
                            
                            ######## Src Reconstruction ###########
                            ref_img_inv=ref_img_inv.repeat(test_batch.shape[0],1,1,1)
                            ref_img_inv=ref_img_inv.to(device)
                            encoder_posterior_ref=model.encode_first_stage(ref_img_inv)
                            z_ref = model.get_first_stage_encoding(encoder_posterior_ref)
                            z2_tar=z2
                            z2=torch.cat([z2_tar,z_ref],dim=0)
                            inverse_cond_inv=torch.cat([inverse_cond,cond_w_src],dim=0)
                            test_model_kwargs_inv=test_model_kwargs.copy()
                            
                            
                            # test_model_kwargs_inv['inpaint_image']=torch.cat([test_model_kwargs['inpaint_image'],test_model_kwargs['inpaint_image']],dim=0)  # just checking
                            # test_model_kwargs_inv['inpaint_mask']=torch.cat([test_model_kwargs['inpaint_mask'],test_model_kwargs['inpaint_mask']],dim=0)  # just checking
                            
                            inpaint_image_src=inpaint_image_src.repeat(test_batch.shape[0],1,1,1)
                            inpaint_mask_src=inpaint_mask_src.repeat(test_batch.shape[0],1,1,1).to(device)
                            test_model_kwargs_inv['inpaint_image']=torch.cat([test_model_kwargs['inpaint_image'],inpaint_image_src ],dim=0)  # just checking
                            test_model_kwargs_inv['inpaint_mask']=torch.cat([test_model_kwargs['inpaint_mask'],inpaint_mask_src],dim=0) 
                            
                            ####################
                            
                            if not os.path.exists(inverse_results_dir_for_batch):
                                os.makedirs(inverse_results_dir_for_batch, exist_ok=True)
                                x_noisy, intermediates = sampler.ddim_invert(x=z2,
                                            cond=inverse_cond_inv,   # what happens if we use c
                                            S=inverse_steps,
                                            shape=shape,
                                            eta=opt.ddim_eta,
                                            unconditional_guidance_scale=opt.scale,
                                            unconditional_conditioning=None,inverse_dir=inverse_results_dir_for_batch,
                                            batch_size=test_batch.shape[0],
                                            test_model_kwargs=test_model_kwargs_inv
                                            )
                                x_noisy = torch.load(os.path.join(inverse_results_dir_for_batch, 'ddim_latents_961.pt'))
                            else:
                                x_noisy = torch.load(os.path.join(inverse_results_dir_for_batch, 'ddim_latents_961.pt'))
                                # start_code=x_noisy.to(x.device)
                            
                            # x_noisy_target,x_noisy_src=x_noisy.chunk(2,dim=0)
                            
                            start_code=x_noisy.to(x.device)
                            # start_code=start_code1
                            video1=test_batch.clone()
                            video1 = F.interpolate(video1, size=(opt.H // opt.f, opt.W // opt.f), mode='bilinear', align_corners=False)
                            # resize to 64,64
                            # video1 = F.interpolate(video1, size=(opt.H // opt.f, opt.W // opt.f), mode='bilinear', align_corners=False)
                            flow= return_flow(video1)
                            
                            # start_code=AdaIn_fusion(x_noisy_target,x_noisy_src,alpha=1.0,beta=0.8,normalized=True)
                            # start_code=fft_fusion(x_noisy_target,x_noisy_src,center=3,center_exclude=1)
                            # start_code= batch_flow_align_latent(
                            #         x_prev=start_code,
                            #         x_prev_recon=z2_tar,
                            #         decode_fn=model.decode_first_stage,# or appropriate decoder
                            #         encode_fn= model.encode_first_stage,
                            #         first_stage_fn=model.get_first_stage_encoding,
                            #         alpha=0.0  # control temporal smoothing
                            #     )
                            
                        elif use_prior:
                            prior=prior.to(device)
                            encoder_posterior_2=model.encode_first_stage(prior)
                            z2 = model.get_first_stage_encoding(encoder_posterior_2)
                            noise = torch.randn_like(z2)
                            x_noisy = model.q_sample(x_start=z2, t=t, noise=noise)
                            start_code = x_noisy
                            # print('start from target')
                        else:
                            noise = torch.randn_like(z)
                            x_noisy = model.q_sample(x_start=z, t=t, noise=noise)
                            if start_code is not None:
                                start_code = x_noisy

                    samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                        conditioning=c,
                                                        target_conditioning=inverse_cond,
                                                        inverse_results_dir=inverse_results_dir_for_batch,
                                                        batch_size=test_batch.shape[0],
                                                        shape=shape,
                                                        verbose=False,
                                                        unconditional_guidance_scale=opt.scale,
                                                        unconditional_conditioning=uc,
                                                        eta=opt.ddim_eta,
                                                        x_T=start_code,
                                                        flow=flow,
                                                        test_model_kwargs=test_model_kwargs,
                                                        src_im=ref_imgs.squeeze(1).to(torch.float32),
                                                        tar=test_batch.to("cuda"))

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                    x_checked_image=x_samples_ddim
                    x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                    if not opt.skip_save:
                        for i,x_sample in enumerate(x_checked_image_torch):

                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            
                            img = Image.fromarray(x_sample.astype(np.uint8)).resize((1024,1024), Image.BILINEAR)
                            img.save(os.path.join(model_out_path, segment_id_batch[i]+".png"))
                            
                            orig_image=Image.open(os.path.join(target_frames_path, str(int(segment_id_batch[i]))+".png"))
                            # To get the consistent output for background just encode and decode
                            image_tensor = get_tensor()(orig_image)
                            image_tensor_resize=transforms.Resize([opt.H, opt.W])(image_tensor)
                            image_tensor_resize=image_tensor_resize.to(device)
                            image_tensor_resize = image_tensor_resize.unsqueeze(0)
                            encoder_posterior = model.encode_first_stage(image_tensor_resize)
                            z = model.get_first_stage_encoding(encoder_posterior)
                            image_tensor_resize=model.decode_first_stage(z)
                            image_tensor_resize = torch.clamp((image_tensor_resize + 1.0) / 2.0, min=0.0, max=1.0)
                            image_tensor_resize = image_tensor_resize.cpu().permute(0, 2, 3, 1).numpy()
                            image_tensor_resize = Image.fromarray((255. * image_tensor_resize[0]).astype(np.uint8)).convert('RGB')
                            image_conv = image_tensor_resize.resize((image_tensor.shape[1], image_tensor.shape[2]), Image.BILINEAR)
                            
                            inv_transforms=inv_transforms_all[int(segment_id_batch[i])]

                            if opt.only_target_crop:                
                                inv_trans_coeffs = inv_transforms
                                swapped_and_pasted = img.convert('RGBA')
                                pasted_image = image_conv.convert('RGBA')
                                swapped_and_pasted.putalpha(255)
                                projected = swapped_and_pasted.transform(orig_image.size, Image.PERSPECTIVE, inv_trans_coeffs, Image.BILINEAR)
                                pasted_image.alpha_composite(projected)
                            
                            # save pasted image
                            pasted_image.save(os.path.join(result_path, segment_id_batch[i]+".png"))
                        
                            base_count += 1

                    if not opt.skip_grid:
                        all_samples.append(x_checked_image_torch)

    path = os.path.join(result_path, '*.png')
    image_filenames = sorted(glob(path))

    clips = ImageSequenceClip(image_filenames, fps=fps)
    name = os.path.basename(out_video_filepath)


    if not no_audio:
        clips = clips.set_audio(video_audio_clip)

    
    clips.write_videofile(out_video_filepath,fps=fps, codec='libx264', audio_codec='aac', ffmpeg_params=['-pix_fmt:v', 'yuv420p', '-colorspace:v', 'bt709', '-color_primaries:v', 'bt709','-color_trc:v', 'bt709', '-color_range:v', 'tv', '-movflags', '+faststart'],logger=proglog.TqdmProgressBarLogger(print_messages=False))

    # sace clips as gif
    
    clips.write_gif(os.path.join(outpath, name.replace('.mp4', '.gif')), fps=fps, logger=proglog.TqdmProgressBarLogger(print_messages=False))

    path = os.path.join(target_frames_path, '*.png')
    
    image_filenames = sorted(glob(path))
    clips = ImageSequenceClip(image_filenames, fps=fps)
    clips.write_videofile(os.path.join(outpath, target_video_name+'_target.mp4'), fps=fps, codec='libx264', audio_codec='aac', ffmpeg_params=['-pix_fmt:v', 'yuv420p', '-colorspace:v', 'bt709',
    '-color_primaries:v', 'bt709','-color_trc:v', 'bt709', '-color_range:v', 'tv', '-movflags', '+faststart'],logger=proglog.TqdmProgressBarLogger(print_messages=False))
    clips.write_gif(os.path.join(outpath, target_video_name+'_target.gif'), fps=fps, logger=proglog.TqdmProgressBarLogger(print_messages=False))
    print('\nDone! {}'.format(out_video_filepath))

    print(f"Your samples are ready and waiting for you here: \n{out_video_filepath} \n"
          f" \nEnjoy.")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a photograph of an astronaut riding a horse",
        help="the prompt to render"
    )

    parser.add_argument(
        "--data_config",
        type=str,
        help="path to data config",
    )

    # parser.add_argument(
    #     "--outdir",
    #     type=str,
    #     nargs="?",
    #     help="dir to write results to",
    #     default="results_video_new/debug"
    # )
    parser.add_argument(
        "--Base_dir",
        type=str,
        nargs="?",
        help="dir to write cropped_images",
        default="results_video_new"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
        default="True"
    )
    parser.add_argument(
        "--Start_from_target",
        action='store_true',
        help="if enabled, uses the noised target image as the starting ",
        default=True
    )
    parser.add_argument(
        "--only_target_crop",
        action='store_true',
        help="if enabled, uses the noised target image as the starting ",
        default=True
    )
    parser.add_argument(
        "--target_start_noise_t",
        type=int,
        default=1000,
        help="target_start_noise_t",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=6,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_frames",
        type=int,
        default=24,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=3.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    # parser.add_argument(
    #     "--target_video",
    #     type=str,
    #     help="target_video",
    #     default="/egr/research-sprintai/baliahsa/mbz-back/Video_diffusion/AnyV2V/data/Data/VFHQ-Test/GT/Vid_Interval1_512x512_LANCZOS4/Clip+-1Jouc19Ixo+P0+C1+F4196-4320/vid.mp4",
    # )
    # parser.add_argument(
    #     "--src_image",
    #     type=str,
    #     help="src_image",
    #     default="/egr/research-sprintai/baliahsa/mbz-back/Video_diffusion/AnyV2V/data/Data/VFHQ-Test/Celeb_Source/10.jpg"
    # )
    parser.add_argument(
        "--src_image_mask",
        type=str,
        help="src_image_mask",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="models/Paint-by-Example/v5_Two_CLIP_proj_154/checkpoints/project_ffhq.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/Paint-by-Example/V5_without_FSA_154/checkpoints/epoch=000019.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    
    parser.add_argument('--faceParser_name', default='default', type=str, help='face parser name, [ default | segnext] is currently supported.')
    parser.add_argument('--faceParsing_ckpt', type=str, default="Other_dependencies/face_parsing/79999_iter.pth")  
    parser.add_argument('--segnext_config', default='', type=str, help='Path to pre-trained SegNeXt faceParser configuration file, '
                                                                        'this option is valid when --faceParsing_ckpt=segenext')

    parser.add_argument('--video_base_dir', type=str, help='base dir for target videos')  
    parser.add_argument('--image_dir', type=str, help='base dir for source images')
    parser.add_argument('--output_base_dir', type=str, help='base dir for output videos')

    parser.add_argument('--save_vis', action='store_true')
    parser.add_argument('--seg12',default=True, action='store_true')
    parser.add_argument('--source_images_dir', type=str, default='data/source_images', help='directory containing source images for inference')
    
    opt = parser.parse_args()

    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    # Load YAML file
    # with open(opt.data_config, "r") as f:
    #     pairs = yaml.safe_load(f)

    ori_base_dir = opt.Base_dir
    all_source_images = os.listdir(opt.source_images_dir)
    video_dirs=os.listdir(opt.video_base_dir)
    
    
    for source_img in all_source_images:
        for video_dir in video_dirs:
            try:
                full_dir = os.path.join(opt.video_base_dir, video_dir)
                video_files = glob(os.path.join(full_dir, "*.mp4"))
                video_file = video_files[0] if video_files else None
                if video_file is not None:
                    opt.target_video = video_file
                    source_name= os.path.basename(source_img)
                    opt.outdir = os.path.join(opt.output_base_dir, video_dir,source_name)
                    # if exists
                    if os.path.exists(opt.outdir):
                        print(f"Output directory {opt.outdir} already exists. Skipping...")
                        continue
                    opt.src_image = os.path.join(opt.image_dir, source_img)
                    opt.Base_dir = os.path.join(ori_base_dir, video_dir)

                    run_inference(
                        model,
                        sampler,
                        opt,
                        device,
                        config,
                    )
            
                else:
                    print(f"Video file not found in {full_dir}")
            except Exception as e:
                print(f"Error processing {video_dir}: {e}")


if __name__ == "__main__":
    main()
