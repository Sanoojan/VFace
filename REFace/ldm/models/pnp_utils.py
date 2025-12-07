import glob

import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import torch
import yaml

import torchvision.transforms as T
from torchvision.io import read_video, write_video
import os
import random
import numpy as np

import logging
logger = logging.getLogger(__name__)


from ldm.modules.attention import exists, default
from einops import rearrange, repeat
from torch import nn, einsum
from scripts.face_swap_utils import *
from scripts.temporal_flow import *

# PNP injection functions
# Modified from ResnetBlock2D.forward
# Modified from models/resnet.py
# from diffusers.utils import USE_PEFT_BACKEND
# from diffusers.models.upsampling import Upsample2D
# from diffusers.models.downsampling import Downsample2D


def find_all_modules_by_name(model,mod_name):
    modules = []
    mod_names=[]
    for name, module in model.named_modules():
        if name.endswith(mod_name):
            modules.append((module))
            mod_names.append(name)
    return modules,mod_names

# Modified from tokenflow_utils.py
def register_time(model, t):
    conv_module = model.unet.up_blocks[1].resnets[1]
    setattr(conv_module, "t", t)
    up_res_dict = {1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    for res in up_res_dict:
        for block in up_res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1.processor
            setattr(module, "t", t)
            module = model.unet.up_blocks[res].temp_attentions[block].transformer_blocks[0].attn1.processor
            setattr(module, "t", t)




def register_spa_attn_injection(model, injection_schedule,switch_on=True,input_blocks=False,output_blocks=True,middle_block=False,attn_component='attn1',chunks=3,flow=None,block_indices=None, fusion="replace",split_ratio_fft=0.8,alpha=0.8):
    
    def temporal_attention(x, window_size=5, sigma=1.0):
        """
        Args:
            x: Tensor of shape (T, C, H, W) — a sequence of frames.
            window_size: Number of frames to consider (must be odd).
            sigma: Standard deviation of the Gaussian kernel.
        Returns:
            output: Gaussian-weighted average tensor of shape (T, C, H, W)
        """
        T = x.shape[0]
        pad = window_size // 2
        output = torch.zeros_like(x)

        # Create Gaussian weights
        offsets = torch.arange(-pad, pad + 1, dtype=torch.float32)
        gauss_kernel = torch.exp(-0.5 * (offsets / sigma) ** 2)
        gauss_kernel /= gauss_kernel.sum()  # Normalize

        for t in range(T):
            weighted_sum = 0.0
            weight_total = 0.0

            for i, offset in enumerate(offsets):
                idx = t + int(offset.item())
                if 0 <= idx < T:
                    weight = gauss_kernel[i]
                    weighted_sum += weight * x[idx]
                    weight_total += weight

            output[t] = weighted_sum / weight_total  # (Optional: should already be normalized)

        return output
    
    def spa_attn_forward(self):
        
        def forward( x, context=None, mask=None,feature_transfer=True):
        
            if feature_transfer:
                batch_size=x.shape[0]
                if switch_on==False:
                    feature_transfer=False  # justt for debugging later code properly
                    
                chunk_size=batch_size//chunks
            
            
            h = self.heads

            q = self.to_q(x)        # 2,4096,320
            context = default(context, x) #2,4096,320
            # if context.shape[-1]==768*2:
            #     # this is for different attention heads
            #     context1,context2=torch.chunk(context,2,dim=-1) # clip/id context1, landmark context2
            #     k1=self.to_k(context1)
            #     k2=self.to_k(context2)
            #     v1=self.to_v(context1)
            #     v2=self.to_v(context2)
                
            #     k=torch.cat([k1[:,:,:self.head_splits[0]*self.dim_head],k2[:,:,-self.head_splits[1]*self.dim_head:]],dim=-1)
            #     v=torch.cat([v1[:,:,:self.head_splits[0]*self.dim_head],v2[:,:,-self.head_splits[1]*self.dim_head:]],dim=-1)
            #     # head_splits=[6,2]
            #     # k1 = self.to_k[0](context1)
            #     # v1 = self.to_v[0](context1)
            #     # k2 = self.to_k[1](context2)
            #     # v2 = self.to_v[1](context2)
            #     # k=torch.cat([k1,k2],dim=-1)
            #     # v=torch.cat([v1,v2],dim=-1)
                
            # else:
            k = self.to_k(context)
            v = self.to_v(context)
            if feature_transfer:
                
                if chunks==3:
                    
                    if fusion=="replace":
                        
                        
                        q[chunk_size:2*chunk_size]=q[:chunk_size]
                        k[chunk_size:2*chunk_size]=k[:chunk_size]
                        # v[chunk_size:2*chunk_size]=v[:chunk_size]
                        
                        q[2*chunk_size:]=q[:chunk_size]
                        k[2*chunk_size:]=k[:chunk_size]
                        # v[2*chunk_size:]=v[:chunk_size]
                        print('pnp feature transfering by replace')
                        
                    elif fusion=="temporal":
                        # q[:chunk_size]  --> Target Video, q[chunk_size:2*chunk_size] --> swapping, q[2*chunk_size:] --> unconditional
                        temp1 = temporal_attention(q[:chunk_size])
                        temp2 = temporal_attention(k[:chunk_size])

                        q[chunk_size:2*chunk_size]= temp1
                        k[chunk_size:2*chunk_size]= temp2
                        
                        q[2*chunk_size:]= temp1
                        k[2*chunk_size:]= temp2
                    elif fusion=="adaIn":
                        q[chunk_size:2*chunk_size]=AdaIn_fusion_for_attn(q[:chunk_size],q[chunk_size:2*chunk_size],alpha=0.9)  
                        k[chunk_size:2*chunk_size]=AdaIn_fusion_for_attn(k[:chunk_size],k[chunk_size:2*chunk_size],alpha=0.9)
                        
                        q[2*chunk_size:]=AdaIn_fusion_for_attn(q[:chunk_size],q[2*chunk_size:],alpha=0.9)
                        k[2*chunk_size:]=AdaIn_fusion_for_attn(k[:chunk_size],k[2*chunk_size:],alpha=0.9)
                    elif fusion=="mix":
                        q[chunk_size:2*chunk_size]=mix_source_and_target(q[:chunk_size],q[chunk_size:2*chunk_size],alpha=0.5)
                        k[chunk_size:2*chunk_size]=mix_source_and_target(k[:chunk_size],k[chunk_size:2*chunk_size],alpha=0.5)
                        
                        q[2*chunk_size:]=mix_source_and_target(q[:chunk_size],q[2*chunk_size:],alpha=0.5)
                        k[2*chunk_size:]=mix_source_and_target(k[:chunk_size],k[2*chunk_size:],alpha=0.5)
                        
                        # print('pnp feature transfering')
                    elif fusion == "fft":
                        # print(q.shape)
                        # q[chunk_size:2*chunk_size]=q[:chunk_size]
                        # k[chunk_size:2*chunk_size]=k[:chunk_size]
                        # # v[chunk_size:2*chunk_size]=v[:chunk_size]
                        
                        # q[2*chunk_size:]=q[:chunk_size]
                        # k[2*chunk_size:]=k[:chunk_size]
                        
                        # breakpoint()
                        q[chunk_size:2*chunk_size]=combine_fft_high_low(q[:chunk_size],q[chunk_size:2*chunk_size],split_ratio=split_ratio_fft)
                        k[chunk_size:2*chunk_size]=combine_fft_high_low(k[:chunk_size],k[chunk_size:2*chunk_size],split_ratio=split_ratio_fft)
                        
                        q[2*chunk_size:]=combine_fft_high_low(q[:chunk_size],q[2*chunk_size:],split_ratio=split_ratio_fft)
                        k[2*chunk_size:]=combine_fft_high_low(k[:chunk_size],k[2*chunk_size:],split_ratio=split_ratio_fft)
                        
                    elif fusion == "flow_fix":
                        # print(q.shape)
                        # q[chunk_size:2*chunk_size]=q[:chunk_size]
                        # k[chunk_size:2*chunk_size]=k[:chunk_size]
                        # # v[chunk_size:2*chunk_size]=v[:chunk_size]
                        
                        # q[2*chunk_size:]=q[:chunk_size]
                        # k[2*chunk_size:]=k[:chunk_size]
                        
                        #Uncomment this to switch on FSAI
                        q[chunk_size:2*chunk_size]=combine_fft_high_low(q[:chunk_size],q[chunk_size:2*chunk_size],split_ratio=split_ratio_fft)
                        k[chunk_size:2*chunk_size]=combine_fft_high_low(k[:chunk_size],k[chunk_size:2*chunk_size],split_ratio=split_ratio_fft)
                        
                        q[2*chunk_size:]=combine_fft_high_low(q[:chunk_size],q[2*chunk_size:],split_ratio=split_ratio_fft)
                        k[2*chunk_size:]=combine_fft_high_low(k[:chunk_size],k[2*chunk_size:],split_ratio=split_ratio_fft)
                        
                        if flow is not None and q.shape[1]==4096:
                            

                            
                            # B,H,w=q.shape
                            q_flow=q[chunk_size:2*chunk_size]
                            k_flow=k[chunk_size:2*chunk_size]
                            q_flow=q_flow.reshape(chunk_size,64,64,-1)
                            k_flow=k_flow.reshape(chunk_size,64,64,-1)
                            q_flow=q_flow.permute(0,3,1,2) # b,c,h,w
                            k_flow=k_flow.permute(0,3,1,2) # b,c,h,w
                            q_flow=align_by_flow(q_flow,flow=flow,alpha=alpha)
                            k_flow=align_by_flow(k_flow,flow=flow,alpha=alpha)
                            q_flow=q_flow.permute(0,2,3,1).reshape(-1,64*64,q_flow.shape[1]) # b,n,c
                            k_flow=k_flow.permute(0,2,3,1).reshape(-1,64*64,k_flow.shape[1]) # b,n,c
                            
                            q[chunk_size:2*chunk_size]=q_flow
                            k[chunk_size:2*chunk_size]=k_flow
                            
                            # q[2*chunk_size:]=q_flow
                            # k[2*chunk_size:]=k_flow
                            print('pnp feature transfering by flow')
                        

                        # else:
                        #     q[chunk_size:2*chunk_size]=q[:chunk_size]
                        #     k[chunk_size:2*chunk_size]=k[:chunk_size]
                        #     # v[chunk_size:2*chunk_size]=v[:chunk_size]
                            
                        #     q[2*chunk_size:]=q[:chunk_size]
                        #     k[2*chunk_size:]=k[:chunk_size]
                        
                        
                        
                        
                        
                    
                    elif fusion == "fft_vfixed":
                        # print(q.shape)
                        # q[chunk_size:2*chunk_size]=q[:chunk_size]
                        # k[chunk_size:2*chunk_size]=k[:chunk_size]
                        # # v[chunk_size:2*chunk_size]=v[:chunk_size]
                        
                        # q[2*chunk_size:]=q[:chunk_size]
                        # k[2*chunk_size:]=k[:chunk_size]
                        
                        
                        q[chunk_size:2*chunk_size]=combine_fft_high_low(q[:chunk_size],q[chunk_size:2*chunk_size],split_ratio=0.8)
                        k[chunk_size:2*chunk_size]=combine_fft_high_low(k[:chunk_size],k[chunk_size:2*chunk_size],split_ratio=0.8)
                        
                        q[2*chunk_size:]=combine_fft_high_low(q[:chunk_size],q[2*chunk_size:],split_ratio=0.8)
                        k[2*chunk_size:]=combine_fft_high_low(k[:chunk_size],k[2*chunk_size:],split_ratio=0.8)

                        # v[:chunk_size]=v[0].unsqueeze(0).repeat(chunk_size,1,1) # repeat the v for all chunks
                        v[chunk_size:2*chunk_size]=v[chunk_size].repeat(chunk_size,1,1) 
                        v[2*chunk_size:]=v[2*chunk_size].repeat(chunk_size,1,1) 
                        # v=v[0].repeat(chunks,1,1) # repeat the v for all chunks
                    
                elif chunks==2:
                    print('pnp feature transfering at inv tar to src')
                    q[chunk_size:]=q[:chunk_size]
                    k[chunk_size:]=k[:chunk_size]
                    
                    # print('pnp feature transfering at src to tar')
                    # q[:chunk_size]=q[chunk_size:]
                    # k[:chunk_size]=k[chunk_size:]
             
            
                    
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
            # print("q.shape",q.shape)

            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
            # print("sim.shape",sim.shape)

            if exists(mask):
                mask = rearrange(mask, 'b ... -> b (...)')
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, 'b j -> (b h) () j', h=h)
                sim.masked_fill_(~mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)

            out = einsum('b i j, b j d -> b i d', attn, v)
            out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
            return self.to_out(out)
        return forward
    if input_blocks:
        spa_attn_modules,module_names = find_all_modules_by_name(model.model.model.diffusion_model.input_blocks,attn_component) 
        processed_modules = []
        for i,module in enumerate(spa_attn_modules):
            if block_indices is None:
                module.forward = spa_attn_forward(module)
                processed_modules.append(module_names[i])
                continue
            elif i in block_indices:
                module.forward = spa_attn_forward(module)
                processed_modules.append(module_names[i])
                continue
                # module.processor.injection_schedule = injection_schedule
                # print(module.processor.injection_schedule)
            
            setattr(module, "injection_schedule", injection_schedule)
        print(processed_modules, "are registered with injection and switched on:",switch_on)
    if output_blocks:
        spa_attn_modules,module_names = find_all_modules_by_name(model.model.model.diffusion_model.output_blocks,attn_component) 
        processed_modules = []
        for i,module in enumerate(spa_attn_modules):
            if block_indices is None:
                module.forward = spa_attn_forward(module)
                processed_modules.append(module_names[i])
                continue
            elif i in block_indices:
                module.forward = spa_attn_forward(module)
                processed_modules.append(module_names[i])
                continue
                # module.processor.injection_schedule = injection_schedule
                # print(module.processor.injection_schedule)
            
            setattr(module, "injection_schedule", injection_schedule)
        print(processed_modules, "are registered with injection and switched on:",switch_on)
    if middle_block:
        spa_attn_modules,module_names = find_all_modules_by_name(model.model.model.diffusion_model.middle_block,attn_component) 
        processed_modules = []
        for i,module in enumerate(spa_attn_modules):
            if block_indices is None:
                module.forward = spa_attn_forward(module)
                processed_modules.append(module_names[i])
                continue
            elif i in block_indices:
                module.forward = spa_attn_forward(module)
                processed_modules.append(module_names[i])
                continue
                # module.processor.injection_schedule = injection_schedule
                # print(module.processor.injection_schedule)
            
            setattr(module, "injection_schedule", injection_schedule)
        print(processed_modules, "are registered with injection and switched on:",switch_on)

def register_conv_injection(model, injection_schedule):
    
    def conv_forward(self):
        def _forward( x, emb):
            if self.updown:
                in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
                h = in_rest(x)
                h = self.h_upd(h)
                x = self.x_upd(x)
                h = in_conv(h)
            else:
                h = self.in_layers(x)
            emb_out = self.emb_layers(emb).type(h.dtype)
            while len(emb_out.shape) < len(h.shape):
                emb_out = emb_out[..., None]
            if self.use_scale_shift_norm:
                out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
                scale, shift = torch.chunk(emb_out, 2, dim=1)
                h = out_norm(h) * (1 + scale) + shift
                h = out_rest(h)
            else:
                h = h + emb_out
                h = self.out_layers(h)
                
            # if self.injection_schedule is not None and (self.t in self.injection_schedule or self.t == 1000):
            if True:
                # logger.debug(f"PnP Injecting Conv at t={self.t}")
                print(f"PnP Injecting Conv")
                
                source_batch_size = int(h.shape[0] // 3)
                # inject unconditional
                h[ :  source_batch_size] = h[2 * source_batch_size :]
                # inject conditional
                h[source_batch_size : 2 * source_batch_size] = h[2 * source_batch_size :]    
            
            return self.skip_connection(x) + h
        
        return _forward
    
    conv_module = model.model.model.diffusion_model.output_blocks[3][0]
    
    conv_module._forward = conv_forward(conv_module)
    # setattr(conv_module, "injection_schedule", injection_schedule)
    
    
    # def conv_forward(self):
    #     def forward(
    #         input_tensor: torch.FloatTensor,
    #         temb: torch.FloatTensor,
    #         scale: float = 1.0,
    #     ) -> torch.FloatTensor:
    #         hidden_states = input_tensor

    #         hidden_states = self.norm1(hidden_states)
    #         hidden_states = self.nonlinearity(hidden_states)

    #         if self.upsample is not None:
    #             # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
    #             if hidden_states.shape[0] >= 64:
    #                 input_tensor = input_tensor.contiguous()
    #                 hidden_states = hidden_states.contiguous()
    #             input_tensor = (
    #                 self.upsample(input_tensor, scale=scale)
    #                 if isinstance(self.upsample, Upsample2D)
    #                 else self.upsample(input_tensor)
    #             )
    #             hidden_states = (
    #                 self.upsample(hidden_states, scale=scale)
    #                 if isinstance(self.upsample, Upsample2D)
    #                 else self.upsample(hidden_states)
    #             )
    #         elif self.downsample is not None:
    #             input_tensor = (
    #                 self.downsample(input_tensor, scale=scale)
    #                 if isinstance(self.downsample, Downsample2D)
    #                 else self.downsample(input_tensor)
    #             )
    #             hidden_states = (
    #                 self.downsample(hidden_states, scale=scale)
    #                 if isinstance(self.downsample, Downsample2D)
    #                 else self.downsample(hidden_states)
    #             )

    #         hidden_states = self.conv1(hidden_states, scale) if not USE_PEFT_BACKEND else self.conv1(hidden_states)

    #         if self.time_emb_proj is not None:
    #             if not self.skip_time_act:
    #                 temb = self.nonlinearity(temb)
    #             temb = (
    #                 self.time_emb_proj(temb, scale)[:, :, None, None]
    #                 if not USE_PEFT_BACKEND
    #                 else self.time_emb_proj(temb)[:, :, None, None]
    #             )

    #         if self.time_embedding_norm == "default":
    #             if temb is not None:
    #                 hidden_states = hidden_states + temb
    #             hidden_states = self.norm2(hidden_states)
    #         elif self.time_embedding_norm == "scale_shift":
    #             if temb is None:
    #                 raise ValueError(
    #                     f" `temb` should not be None when `time_embedding_norm` is {self.time_embedding_norm}"
    #                 )
    #             time_scale, time_shift = torch.chunk(temb, 2, dim=1)
    #             hidden_states = self.norm2(hidden_states)
    #             hidden_states = hidden_states * (1 + time_scale) + time_shift
    #         else:
    #             hidden_states = self.norm2(hidden_states)

    #         hidden_states = self.nonlinearity(hidden_states)

    #         hidden_states = self.dropout(hidden_states)
    #         hidden_states = self.conv2(hidden_states, scale) if not USE_PEFT_BACKEND else self.conv2(hidden_states)

    #         if self.injection_schedule is not None and (self.t in self.injection_schedule or self.t == 1000):
    #             logger.debug(f"PnP Injecting Conv at t={self.t}")
    #             source_batch_size = int(hidden_states.shape[0] // 3)
    #             # inject unconditional
    #             hidden_states[source_batch_size : 2 * source_batch_size] = hidden_states[:source_batch_size]
    #             # inject conditional
    #             hidden_states[2 * source_batch_size :] = hidden_states[:source_batch_size]

    #         if self.conv_shortcut is not None:
    #             input_tensor = (
    #                 self.conv_shortcut(input_tensor, scale)
    #                 if not USE_PEFT_BACKEND
    #                 else self.conv_shortcut(input_tensor)
    #             )

    #         output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

    #         return output_tensor

    #     return forward

    # conv_module = model.unet.up_blocks[1].resnets[1]
    # conv_module.forward = conv_forward(conv_module)
    # setattr(conv_module, "injection_schedule", injection_schedule)


# Modified from AttnProcessor2_0.__call__
# Modified from models/attention.py
# from typing import Optional
# from diffusers.models.attention_processor import AttnProcessor2_0

# def register_spatial_attention_pnp(model, injection_schedule):
#     class ModifiedSpaAttnProcessor(AttnProcessor2_0):
#         def __call__(
#             self,
#             attn,  # attn: Attention,
#             hidden_states: torch.FloatTensor,
#             encoder_hidden_states: Optional[torch.FloatTensor] = None,
#             attention_mask: Optional[torch.FloatTensor] = None,
#             temb: Optional[torch.FloatTensor] = None,
#             scale: float = 1.0,
#         ) -> torch.FloatTensor:
#             residual = hidden_states
#             if attn.spatial_norm is not None:
#                 hidden_states = attn.spatial_norm(hidden_states, temb)

#             input_ndim = hidden_states.ndim

#             if input_ndim == 4:
#                 batch_size, channel, height, width = hidden_states.shape
#                 hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

#             batch_size, sequence_length, _ = (
#                 hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
#             )

#             # Modified here
#             chunk_size = batch_size // 3  # batch_size is 3*chunk_size because concat[source, uncond, cond]

#             if attention_mask is not None:
#                 attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
#                 # scaled_dot_product_attention expects attention_mask shape to be
#                 # (batch, heads, source_length, target_length)
#                 attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

#             if attn.group_norm is not None:
#                 hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

#             args = () if USE_PEFT_BACKEND else (scale,)
#             query = attn.to_q(hidden_states, *args)

#             if encoder_hidden_states is None:
#                 encoder_hidden_states = hidden_states
#             elif attn.norm_cross:
#                 encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

#             key = attn.to_k(encoder_hidden_states, *args)
#             value = attn.to_v(encoder_hidden_states, *args)

#             # Modified here.
#             if self.injection_schedule is not None and (self.t in self.injection_schedule or self.t == 1000):
#                 logger.debug(f"PnP Injecting Spa-Attn at t={self.t}")
#                 # inject source into unconditional
#                 query[chunk_size : 2 * chunk_size] = query[:chunk_size]
#                 key[chunk_size : 2 * chunk_size] = key[:chunk_size]
#                 # inject source into conditional
#                 query[2 * chunk_size :] = query[:chunk_size]
#                 key[2 * chunk_size :] = key[:chunk_size]

#             inner_dim = key.shape[-1]
#             head_dim = inner_dim // attn.heads

#             query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

#             key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
#             value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

#             # the output of sdp = (batch, num_heads, seq_len, head_dim)
#             # TODO: add support for attn.scale when we move to Torch 2.1
#             hidden_states = F.scaled_dot_product_attention(
#                 query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
#             )

#             hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
#             hidden_states = hidden_states.to(query.dtype)

#             # linear proj
#             hidden_states = attn.to_out[0](hidden_states, *args)
#             # dropout
#             hidden_states = attn.to_out[1](hidden_states)

#             if input_ndim == 4:
#                 hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

#             if attn.residual_connection:
#                 hidden_states = hidden_states + residual

#             hidden_states = hidden_states / attn.rescale_output_factor

#             return hidden_states

#     # for _, module in model.unet.named_modules():
#     #     if isinstance_str(module, "BasicTransformerBlock"):
#     #         module.attn1.processor.__call__ = sa_processor__call__(module.attn1.processor)
#     #         setattr(module.attn1.processor, "injection_schedule", [])  # Disable PNP

#     res_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
#     # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
#     for res in res_dict:
#         for block in res_dict[res]:
#             module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
#             modified_processor = ModifiedSpaAttnProcessor()
#             setattr(modified_processor, "injection_schedule", injection_schedule)
#             module.processor = modified_processor



# def register_temp_attention_pnp(model, injection_schedule):
#     class ModifiedTmpAttnProcessor(AttnProcessor2_0):
#         def __call__(
#             self,
#             attn,  # attn: Attention,
#             hidden_states: torch.FloatTensor,
#             encoder_hidden_states: Optional[torch.FloatTensor] = None,
#             attention_mask: Optional[torch.FloatTensor] = None,
#             temb: Optional[torch.FloatTensor] = None,
#             scale: float = 1.0,
#         ) -> torch.FloatTensor:
#             residual = hidden_states
#             if attn.spatial_norm is not None:
#                 hidden_states = attn.spatial_norm(hidden_states, temb)

#             input_ndim = hidden_states.ndim

#             if input_ndim == 4:
#                 batch_size, channel, height, width = hidden_states.shape
#                 hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

#             batch_size, sequence_length, _ = (
#                 hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
#             )

#             # Modified here
#             chunk_size = batch_size // 3  # batch_size is 3*chunk_size because concat[source, uncond, cond]

#             if attention_mask is not None:
#                 attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
#                 # scaled_dot_product_attention expects attention_mask shape to be
#                 # (batch, heads, source_length, target_length)
#                 attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

#             if attn.group_norm is not None:
#                 hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

#             args = () if USE_PEFT_BACKEND else (scale,)
#             query = attn.to_q(hidden_states, *args)

#             if encoder_hidden_states is None:
#                 encoder_hidden_states = hidden_states
#             elif attn.norm_cross:
#                 encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

#             key = attn.to_k(encoder_hidden_states, *args)
#             value = attn.to_v(encoder_hidden_states, *args)

#             # Modified here.
#             if self.injection_schedule is not None and (self.t in self.injection_schedule or self.t == 1000):
#                 logger.debug(f"PnP Injecting Tmp-Attn at t={self.t}")
#                 # inject source into unconditional
#                 query[chunk_size : 2 * chunk_size] = query[:chunk_size]
#                 key[chunk_size : 2 * chunk_size] = key[:chunk_size]
#                 # inject source into conditional
#                 query[2 * chunk_size :] = query[:chunk_size]
#                 key[2 * chunk_size :] = key[:chunk_size]

#             inner_dim = key.shape[-1]
#             head_dim = inner_dim // attn.heads

#             query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

#             key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
#             value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

#             # the output of sdp = (batch, num_heads, seq_len, head_dim)
#             # TODO: add support for attn.scale when we move to Torch 2.1
#             hidden_states = F.scaled_dot_product_attention(
#                 query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
#             )

#             hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
#             hidden_states = hidden_states.to(query.dtype)

#             # linear proj
#             hidden_states = attn.to_out[0](hidden_states, *args)
#             # dropout
#             hidden_states = attn.to_out[1](hidden_states)

#             if input_ndim == 4:
#                 hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

#             if attn.residual_connection:
#                 hidden_states = hidden_states + residual

#             hidden_states = hidden_states / attn.rescale_output_factor

#             return hidden_states
#     # for _, module in model.unet.named_modules():
#     #     if isinstance_str(module, "BasicTransformerBlock"):
#     #         module.attn1.processor.__call__ = ta_processor__call__(module.attn1.processor)
#     #         setattr(module.attn1.processor, "injection_schedule", [])  # Disable PNP

#     res_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
#     # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
#     for res in res_dict:
#         for block in res_dict[res]:
#             module = model.unet.up_blocks[res].temp_attentions[block].transformer_blocks[0].attn1
#             modified_processor = ModifiedTmpAttnProcessor()
#             setattr(modified_processor, "injection_schedule", injection_schedule)
#             module.processor = modified_processor