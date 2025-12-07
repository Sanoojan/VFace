"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial
from src.Face_models.encoders.model_irse import Backbone
import torch.nn as nn
import torchvision.transforms.functional as TF
import os
from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, \
    extract_into_tensor
    
from PIL import Image
from ldm.models.pnp_utils import *
from scripts.face_swap_utils import *
from scripts.temporal_flow import batch_flow_align,batch_flow_align_latent,align_by_flow,align_by_flow_high_res




def load_ddim_latents_at_t(t, ddim_latents_path):
    ddim_latents_at_t_path = os.path.join(ddim_latents_path, f"ddim_latents_{t}.pt")
    assert os.path.exists(ddim_latents_at_t_path), f"Missing latents at t {t} path {ddim_latents_at_t_path}"
    ddim_latents_at_t = torch.load(ddim_latents_at_t_path)
    return ddim_latents_at_t



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
    
def save_latent_img(model,latents, path,ind=0):
    img=model.decode_first_stage(latents)
    img = torch.clamp((img + 1.0) / 2.0, min=0.0, max=1.0)
    img = img.cpu().permute(0, 2, 3, 1).numpy()
    img=img[ind]
    img = (img * 255).astype(np.uint8)
    img = Image.fromarray(img)
    img.save(path)
    
def analyse_fft(tar,src,model):
    for i in range(1,64):
        for j in range(i):
            
            save_noise=fft_fusion(tar,src,center=i,center_exclude=j)
            save_latent_img(model,save_noise,path=f"Debug/fft_analysis/comb_{i}_{j}.jpg",ind=4)

class IDLoss(nn.Module):
    def __init__(self,path="Other_dependencies/arcface/model_ir_se50.pth",multiscale=False):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        
        self.multiscale = multiscale
        self.face_pool_1 = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        # self.facenet=iresnet100(pretrained=False, fp16=False) # changed by sanoojan
        
        self.facenet.load_state_dict(torch.load(path))
        
        self.face_pool_2 = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()
        
        self.set_requires_grad(False)
            
    def set_requires_grad(self, flag=True):
        for p in self.parameters():
            p.requires_grad = flag
    
    def extract_feats(self, x,clip_img=True):
        # breakpoint()
        if clip_img:
            x = un_norm_clip(x)
            x = TF.normalize(x, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        x = self.face_pool_1(x)  if x.shape[2]!=256 else  x # (1) resize to 256 if needed
        x = x[:, :, 35:223, 32:220]  # (2) Crop interesting region
        x = self.face_pool_2(x) # (3) resize to 112 to fit pre-trained model
        # breakpoint()
        x_feats = self.facenet(x, multi_scale=self.multiscale )
        
        # x_feats = self.facenet(x) # changed by sanoojan
        return x_feats

    

    def forward(self, y_hat, y,clip_img=True,return_seperate=False):
        n_samples = y.shape[0]
        y_feats_ms = self.extract_feats(y,clip_img=clip_img)  # Otherwise use the feature from there

        y_hat_feats_ms = self.extract_feats(y_hat,clip_img=clip_img)
        y_feats_ms = [y_f.detach() for y_f in y_feats_ms]
        
        loss_all = 0
        sim_improvement_all = 0
        seperate_losses=[]
        for y_hat_feats, y_feats in zip(y_hat_feats_ms, y_feats_ms):
 
            loss = 0
            sim_improvement = 0
            count = 0
            
            for i in range(n_samples):
                sim_target = y_hat_feats[i].dot(y_feats[i])
                sim_views = y_feats[i].dot(y_feats[i])

                seperate_losses.append(1-sim_target)
                loss += 1 - sim_target  # id loss
                sim_improvement +=  float(sim_target) - float(sim_views)
                count += 1
            
            loss_all += loss / count
            sim_improvement_all += sim_improvement / count
    
        return loss_all, sim_improvement_all, None
    

class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        # self.ID_LOSS=IDLoss()

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               target_conditioning=None,
               inverse_results_dir=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               flow=None,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,src_im=None,tar=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')
        
        
        
        samples, intermediates = self.ddim_sampling(conditioning,     
                                                    size,
                                                    target_conditioning=target_conditioning,
                                                    inverse_results_dir=inverse_results_dir,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    flow=flow,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    src_im=src_im,
                                                    **kwargs
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      target_conditioning=None,
                      inverse_results_dir=None,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None,flow=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,src_im=None,**kwargs):
        device = self.model.betas.device
        
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        if src_im is not None:
            src_im=un_norm_clip(src_im)

        # TODO: properly code injection schedule
        # pnp injection
        # switch off pnp injection
        register_spa_attn_injection(self, 1,switch_on=False,input_blocks=True,middle_block=True, output_blocks=True,attn_component="attn1", chunks=3)
        #pnp feature transfer    
        register_spa_attn_injection(self, 1,switch_on=True,input_blocks=False,middle_block=False, output_blocks=True,attn_component="attn1", chunks=3,block_indices=[0,1,2,3,4,5,6,7,8],fusion="fft",split_ratio_fft=0.8,alpha=0.0)
        # register_conv_injection(self, 1) 
        # register_spa_attn_injection(self, 1,switch_on=True,input_blocks=False,middle_block=False, output_blocks=True,attn_component="attn1", chunks=3,block_indices=[6,7],fusion="temporal")
        # register_spa_attn_injection(self, 1,switch_on=True,input_blocks=True,middle_block=False, output_blocks=False,attn_component="attn1",flow=flow, chunks=3,block_indices=[0,1,2,3],fusion="flow_fix")
        
        # register_spa_attn_injection(self, 1,switch_on=True,input_blocks=False,middle_block=False, output_blocks=True,attn_component="attn1",flow=flow, chunks=3,block_indices=[0,1,2,3,4,5,6,7,8],fusion="fft")
        
        # register_spa_attn_injection(self, 1,switch_on=True,input_blocks=False,middle_block=False, output_blocks=True,attn_component="attn1", chunks=3,block_indices=[0,1,2],fusion="fft")
        
        for i, step in enumerate(iterator):
            
            if True:
                register_spa_attn_injection(self, 1,switch_on=False,input_blocks=True,middle_block=True, output_blocks=True,attn_component="attn1",flow=flow, chunks=3,block_indices=[0,1,2,3,4,5,6,7,8],fusion="flow_fix",split_ratio_fft=0.8,alpha=0.0)
                # register_spa_attn_injection(self, 1,switch_on=True,input_blocks=False,middle_block=False, output_blocks=True,attn_component="attn1",flow=flow, chunks=3,block_indices=[8],fusion="flow_fix")
                register_spa_attn_injection(self, 1,switch_on=True,input_blocks=True,middle_block=False, output_blocks=False,attn_component="attn1",flow=flow, chunks=3,block_indices=[0,1,2,3,4,5,6,7,8],fusion="flow_fix",split_ratio_fft=0.8,alpha=0.0)
            else:
                register_spa_attn_injection(self, 1,switch_on=False,input_blocks=True,middle_block=False, output_blocks=True,attn_component="attn1",flow=flow, chunks=3,block_indices=[0,1,2,3,4,5,6,7,8],fusion="flow_fix",split_ratio_fft=0.8,alpha=0.0)    
                register_spa_attn_injection(self, 1,switch_on=True,input_blocks=True,middle_block=True, output_blocks=True,attn_component="attn1",flow=flow, chunks=3,block_indices=[0,1,2,3,4,5,6,7,8],fusion="fft",split_ratio_fft=0.8,alpha=0.0)
            # if i==total_steps//2:
                # register_spa_attn_injection(self, 1,switch_on=False,input_blocks=False,output_blocks=True,attn_component="attn1")
                # register_spa_attn_injection(self, 1,switch_on=True,input_blocks=True,output_blocks=False,attn_component="attn1")
            
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img
            if target_conditioning is not None:
                
                # pnp conv transfer
                src_start=None
                # if i==0:
                #     # create random noise like img
                #     src_start=torch.randn_like(img)
                
                outs = self.p_sample_ddim_with_inverse(img, cond, ts, 
                                        target_conditioning=target_conditioning,
                                        inverse_results_dir=inverse_results_dir,
                                        index=index,src_start=None, use_original_steps=ddim_use_original_steps,
                                        quantize_denoised=quantize_denoised, temperature=temperature,
                                        noise_dropout=noise_dropout, score_corrector=score_corrector,
                                        corrector_kwargs=corrector_kwargs,
                                        unconditional_guidance_scale=unconditional_guidance_scale,flow=flow,
                                        unconditional_conditioning=unconditional_conditioning,src_im=src_im,**kwargs)
            else:
                outs = self.p_sample_ddim(img, cond, ts,
                                        index=index, use_original_steps=ddim_use_original_steps,
                                        quantize_denoised=quantize_denoised, temperature=temperature,
                                        noise_dropout=noise_dropout, score_corrector=score_corrector,
                                        corrector_kwargs=corrector_kwargs,
                                        unconditional_guidance_scale=unconditional_guidance_scale,
                                        unconditional_conditioning=unconditional_conditioning,src_im=src_im,**kwargs)
            
            
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates
    
    

    
    @torch.no_grad()
    def ddim_invert(self, x, cond, S, shape, eta=0., 
                    unconditional_guidance_scale=1., 
                    unconditional_conditioning=None,inverse_dir=None,batch_size=6,src_lm=None,tar_lm=None, **kwargs):
        """
        Perform DDIM inversion to estimate the noise that led to the given image `x`.

        Args:
            x: The observed image to be inverted.
            cond: Conditioning input (e.g., text embeddings).
            S: Number of steps used in sampling.
            shape: Shape of the input tensor.
            eta: DDIM noise parameter.
            unconditional_guidance_scale: Scale for classifier-free guidance.
            unconditional_conditioning: Unconditional conditioning for guidance.

        Returns:
            x_T: The estimated initial noise (x_T).
            intermediates: Intermediate steps from inversion.
        """
        device = x.device
        b = x.shape[0]
        
        
        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=False)
        timesteps = self.ddim_timesteps

        intermediates = {'x_inter': [x]}
        
        register_spa_attn_injection(self, 1,switch_on=False,input_blocks=True,middle_block=True, output_blocks=True,attn_component="attn1", chunks=3)
        # register_spa_attn_injection(self, 1,switch_on=True,input_blocks=True,middle_block=False, output_blocks=True,attn_component="attn1", chunks=2,block_indices=[0,1,2])
        
        for i, step in enumerate(tqdm(timesteps, desc="DDIM Inversion", total=len(timesteps))):
            
            #skip last step
            # if i > len(timesteps) :
            #     continue
            
            # if i>len(timesteps)//2:
            #     register_spa_attn_injection(self, 1,switch_on=False,input_blocks=False,middle_block=False, output_blocks=True,attn_component="attn1", chunks=2)
                
            
            inversion_save_path="Debug/inversions4"
            debug=False
            if debug:
                if not os.path.exists(inversion_save_path):
                    os.makedirs(inversion_save_path)
                # save x
                img=self.model.decode_first_stage(x)
                save_clip_img(img[0], os.path.join(inversion_save_path, f"ddim_inversion_{step}.png"),clip=False)
                
                
                # img=un_norm_clip(img)
                # img = TF.normalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                # img = TF.to_pil_image(img[0].cpu())
                # img.save(os.path.join(inversion_save_path, f"ddim_inversion_{step}.png"))
            
            
            index = i
            ts = torch.full((b,), step, device=device, dtype=torch.long)
            if 'test_model_kwargs' in kwargs:
                kwargs1=kwargs['test_model_kwargs']
                x = torch.cat([x, kwargs1['inpaint_image'], kwargs1['inpaint_mask']],dim=1)
            elif 'rest' in kwargs:
                x = torch.cat((x, kwargs['rest']), dim=1)
            # Predict noise and reconstruct x_T step by step
            if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                e_t = self.model.apply_model(x, ts, cond)
            else:
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([ts] * 2)
                c_in = torch.cat([unconditional_conditioning, cond])
                e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
                e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

            alphas = self.ddim_alphas
            sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
            sigmas = self.ddim_sigmas

            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)

            alpha_next_t=self.alphas_cumprod[step]
            # alpha_next_t=torch.full((b, 1, 1, 1), alpha_next_t, device=device)
            
            current_t=max(0,step-(1000//len(timesteps)))
            alpha_t = self.alphas_cumprod[current_t]
            
            # nosie=(x[:,:4,:,:]-sqrt_one_minus_at*e_t)*alpha_next_t.sqrt()/a_t.sqrt()+(1-alpha_next_t).sqrt()*e_t
            nosie=(x[:,:4,:,:]-(1-alpha_t).sqrt()*e_t)*alpha_next_t.sqrt()/alpha_t.sqrt()+(1-alpha_next_t).sqrt()*e_t
            
            # # Reverse step
            # pred_x0 = (x[:,:4,:,:] - sqrt_one_minus_at * e_t) / a_t.sqrt()
            # nosie = (x[:,:4,:,:] - a_t.sqrt() * pred_x0) / (1. - a_t).sqrt()
            
            # pred_x0 = (x[:,:4,:,:] - (1-alpha_t).sqrt() * e_t) / a_t.sqrt()
            # nosie = (x[:,:4,:,:] - alpha_t.sqrt() * pred_x0) / (1. - alpha_t).sqrt()

            # Store intermediate results
            intermediates['x_inter'].append(nosie)
            
            x = nosie  # Update x to continue inversion
            
            # save_noise= ((nosie[:batch_size]+nosie[batch_size:])/1.41)
            x_noisy_target= nosie[:batch_size]
            x_noisy_src= nosie[batch_size:]
            if i<len(timesteps)//2:
                # save_noise=fft_fusion(x_noisy_target,x_noisy_src,center=17,center_exclude=0)
                # save_noise= AdaIn_fusion(nosie[:batch_size],nosie[batch_size:],alpha=1.0,beta=0.8,normalized=True)
                save_noise=x_noisy_target
                # save_noise=x_noisy_src
            else:
                
                # save_noise= AdaIn_fusion(nosie[:batch_size],nosie[batch_size:],alpha=1.0,beta=0.8,normalized=True)
                # save_noise=fft_fusion(x_noisy_target,x_noisy_src,center=17,center_exclude=0)
                save_noise=x_noisy_target
                # save_noise=x_noisy_src
            # if i == 1:
            #     save_noise_2=fft_fusion_warp(x_noisy_target,x_noisy_src,center=5,center_exclude=3,lm_src=src_lm,lm_tar=tar_lm)
            #     save_latent_img(self.model ,save_noise_2,path=f"Debug/yohan/comb_check.jpg",ind=5)
                
            
            # save noise
            torch.save(
                save_noise.detach().clone(),
                os.path.join(inverse_dir, f"ddim_latents_{step}.pt"),
            )
            
            

        return nosie, intermediates
    
    

    @torch.no_grad()
    def p_sample_ddim_guided(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,src_im=None,**kwargs):
        b, *_, device = *x.shape, x.device
        if 'test_model_kwargs' in kwargs:
            kwargs=kwargs['test_model_kwargs']
            x = torch.cat([x, kwargs['inpaint_image'], kwargs['inpaint_mask']],dim=1)
        elif 'rest' in kwargs:
            x = torch.cat((x, kwargs['rest']), dim=1)
        else:
            raise Exception("kwargs must contain either 'test_model_kwargs' or 'rest' key")
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
        else:  # check @ sanoojan
            x_in = torch.cat([x] * 2) #x_in: 2,9,64,64
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c]) #c_in: 2,1,768
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond) #1,4,64,64

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        if x.shape[1]!=4:
            pred_x0 = (x[:,:4,:,:] - sqrt_one_minus_at * e_t) / a_t.sqrt()
            # G_id=ID_LOSS
            seperate_sim=None
            src_im=None
            if src_im is not None:
                pred_x0_im=self.model.decode_first_stage(pred_x0)
                masks=1-TF.resize(x[:,8,:,:],(pred_x0_im.shape[2],pred_x0_im.shape[3]))
                #mask x_samples_ddim
                pred_x0_im_masked=pred_x0_im*masks.unsqueeze(1)
                # x_samples_ddim_masked=un_norm_clip(x_samples_ddim_masked)
                # x_samples_ddim_masked = TF.normalize(x_samples_ddim_masked, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                
                ID_loss,_,seperate_sim=self.model.face_ID_model(pred_x0_im_masked,src_im,clip_img=False,return_seperate=True)
                grad=torch.autograd.grad(ID_loss,x)[0]
        else:
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
            
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(dir_xt.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        if seperate_sim is None:
            x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        else:  
            seperate_sim=3*torch.tensor(seperate_sim)
            #make upper limit 1 and lower limit 0
            seperate_sim=torch.clamp(seperate_sim,0,1)
            x_prev = a_prev.sqrt() * pred_x0 + seperate_sim.view(-1,1,1,1).to(self.model.device)*dir_xt + noise
        return x_prev, pred_x0
    
    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index,repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,**kwargs):
        b, *_, device = *x.shape, x.device
        
        if 'test_model_kwargs' in kwargs:
            kwargs=kwargs['test_model_kwargs']
            x = torch.cat([x, kwargs['inpaint_image'], kwargs['inpaint_mask']],dim=1)
        elif 'rest' in kwargs:
            x = torch.cat((x, kwargs['rest']), dim=1)
        else:
            raise Exception("kwargs must contain either 'test_model_kwargs' or 'rest' key")
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
        else:  # check @ sanoojan
            x_in = torch.cat([x] * 2) #x_in: 2,9,64,64
            t_in = torch.cat([t] * 2)
            # if self.model.stack_feat:
            #     e_t_uncond=self.model.apply_model(x, t, unconditional_conditioning)
            #     e_t = self.model.apply_model(x, t, c)
            # else:
            c_in = torch.cat([unconditional_conditioning, c]) #c_in: 2,1,768
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond) #1,4,64,64

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        if x.shape[1]!=4:
            pred_x0 = (x[:,:4,:,:] - sqrt_one_minus_at * e_t) / a_t.sqrt()
        else:
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(dir_xt.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0
    

    
    @torch.no_grad()
    def p_sample_ddim_with_inverse(self, x, c, t, index, target_conditioning=None,
                      inverse_results_dir=None,repeat_noise=False,src_start=None, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1.,flow=None, unconditional_conditioning=None,**kwargs):
        b, *_, device = *x.shape, x.device
        
        ddim_inv_t = load_ddim_latents_at_t(t[0].item(), inverse_results_dir).to(x.device)
        
        
        if 'test_model_kwargs' in kwargs:
            kwargs=kwargs['test_model_kwargs']
            x = torch.cat([x, kwargs['inpaint_image'], kwargs['inpaint_mask']],dim=1)
            if src_start is not None:
                x_uncond = torch.cat([src_start, kwargs['inpaint_image'], kwargs['inpaint_mask']],dim=1)
            else:
                x_uncond=x
            ddim_inv_t = torch.cat([ddim_inv_t, kwargs['inpaint_image'], kwargs['inpaint_mask']],dim=1)
            
        elif 'rest' in kwargs:
            x = torch.cat((x, kwargs['rest']), dim=1)
            if src_start is not None:
                x_uncond = torch.cat([src_start, kwargs['rest']], dim=1)
            else:
                x_uncond = x
            
            ddim_inv_t = torch.cat((ddim_inv_t, kwargs['rest']), dim=1)
        else:
            raise Exception("kwargs must contain either 'test_model_kwargs' or 'rest' key")
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
            
        else:  # check @ sanoojan
            x_in = torch.cat([x,x_uncond]) #x_in: 2,9,64,64
            x_in=torch.cat([x_in,ddim_inv_t],dim=0)
            t_in = torch.cat([t] * 3)
            # if self.model.stack_feat:
            #     e_t_uncond=self.model.apply_model(x, t, unconditional_conditioning)
            #     e_t = self.model.apply_model(x, t, c)
            # else:
            c_in = torch.cat([unconditional_conditioning, c]) #c_in: 2,1,768
            c_in=torch.cat([c_in, target_conditioning],dim=0)
            
            e_t_uncond, e_t, e_t_recon = self.model.apply_model(x_in, t_in, c_in).chunk(3)
            
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond) #1,4,64,64
            e_t_recon = e_t_recon + unconditional_guidance_scale * (e_t_recon - e_t_uncond) #1,4,64,64

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)
            e_t_recon = score_corrector.modify_score(self.model, e_t_recon, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        if x.shape[1]!=4:
            pred_x0 = (x[:,:4,:,:] - sqrt_one_minus_at * e_t) / a_t.sqrt()
            pred_x0_recon = (ddim_inv_t[:,:4,:,:] - sqrt_one_minus_at * e_t_recon) / a_t.sqrt()
            
        else:
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
            pred_x0_recon = (ddim_inv_t - sqrt_one_minus_at * e_t_recon) / a_t.sqrt()   
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
            pred_x0_recon, _, *_ = self.model.first_stage_model.quantize(pred_x0_recon)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(dir_xt.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        
        
        dir_xt_recon = (1. - a_prev - sigma_t**2).sqrt() * e_t_recon
        noise_recon = sigma_t * noise_like(dir_xt_recon.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:  
            noise_recon = torch.nn.functional.dropout(noise_recon, p=noise_dropout)
        x_prev_recon = a_prev.sqrt() * pred_x0_recon + dir_xt_recon + noise_recon
        
        # if t[0].item()>130 and t[0].item()<150:
        #     print(t)
            
        #     # x_prev= align_by_flow(x_prev=x_prev,flow=flow,alpha=0.5)
        #     x_prev= align_by_flow_high_res(x_prev=x_prev,flow=flow,decode_fn=self.model.decode_first_stage,
        #                                   encode_fn= self.model.encode_first_stage,
        #                                   first_stage_fn=self.model.get_first_stage_encoding,
        #                                   alpha=0.5)
            
            
            # x_prev = batch_flow_align(
            #     x_prev=x_prev,
            #     x_prev_recon=x_prev_recon,
            #     decode_fn=self.model.decode_first_stage,# or appropriate decoder
            #     encode_fn= self.model.encode_first_stage,
            #     first_stage_fn=self.model.get_first_stage_encoding,
            #     alpha=0.0  # control temporal smoothing
            # )
            
            # x_prev = batch_flow_align_latent(
            #     x_prev=x_prev,
            #     x_prev_recon=x_prev_recon,
            #     decode_fn=self.model.decode_first_stage,# or appropriate decoder
            #     encode_fn= self.model.encode_first_stage,
            #     first_stage_fn=self.model.get_first_stage_encoding,
            #     alpha=0.0  # control temporal smoothing
            # )
        
        
        return x_prev, pred_x0
    
    
    
    def sample_train(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               t=None,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')
        # for param in self.model.first_stage_model.parameters():
        #     param.requires_grad = False
        samples, intermediates = self.ddim_sampling_train(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,ddim_num_steps=S,
                                                    curr_t=t,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    **kwargs
                                                    )
        return samples, intermediates

 
    def ddim_sampling_train(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,ddim_num_steps=None,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,curr_t=None,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,**kwargs):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T
            
        kwargs['rest']=img[:,4:,:,:]
        img=img[:,:4,:,:]
        

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        curr_t=curr_t.cpu().numpy()
        skip = (curr_t-1) // ddim_num_steps
        # replace all 0s with 1s
        skip[skip == 0] = 1
        if type(skip)!=int:
            seq=[range(1, curr_t[n]-1, skip[n]) for n in range(len(curr_t))]
            min_length = min(len(sublist) for sublist in seq)
            min_length=min(min_length,ddim_num_steps)
            # Create a new list of sublists by truncating each sublist to the minimum length
            truncated_seq = [sublist[:min_length] for sublist in seq]
            seq= np.array(truncated_seq)

            # seq=np.flip(seq)
        #concatenate all sequences
        # seq = np.concatenate(seq)
        seq=torch.from_numpy(seq).to(device)
        seq=torch.flip(seq,dims=[1])

        
        
        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        # time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        # total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        # print(f"Running DDIM Sampling with {total_steps} timesteps")


        # time_range=np.array([1])
        # iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        
        total_steps=seq.shape[1]
        for i in range(seq.shape[1]):
            index = total_steps - i - 1
            # ts = torch.full((b,), step, device=device, dtype=torch.long)
            ts=seq[:,i].long()
            #make it toech long
            # ts=ts.long()

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img
            outs = self.p_sample_ddim_train(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,**kwargs)
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    
    def p_sample_ddim_train(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,return_features=False,**kwargs):
        b, *_, device = *x.shape, x.device
        # if 'test_model_kwargs' in kwargs:
        #     kwargs=kwargs['test_model_kwargs']
        #     x = torch.cat([x, kwargs['inpaint_image'], kwargs['inpaint_mask']],dim=1)
        if 'rest' in kwargs:
            x = torch.cat((x, kwargs['rest']), dim=1)
    
            
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c,return_features=return_features)
        else:  # check @ sanoojan
            x_in = torch.cat([x] * 2) #x_in: 2,9,64,64
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c]) #c_in: 2,1,768
            if return_features:
                e_t_uncond, e_t,features = self.model.apply_model(x_in, t_in, c_in,return_features=return_features).chunk(3)
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond) #1,4,64,64

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        if x.shape[1]!=4:
            pred_x0 = (x[:,:4,:,:] - sqrt_one_minus_at * e_t) / a_t.sqrt()
        else:
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(dir_xt.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0
    

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
        return x_dec