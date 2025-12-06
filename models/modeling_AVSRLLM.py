#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 23:09:38 2024

@author: umbertocappellazzo
"""
import sys
sys.path.append("..")
import torch
from torch import nn

from .Llama_LoRA import LlamaForCausalLM_lora
from transformers import WhisperModel, LlamaForCausalLM, AutoFeatureExtractor, WavLMModel

import fairseq
from av_hubert.avhubert.hubert_asr import AVHubertSeq2Seq, AVHubertSeq2SeqConfig
from av_hubert.avhubert.hubert_lora import AVHubertModel_lora
from .uadf_block import UADFBlock
import math

IGNORE_INDEX = -100

class AVSR_LLMs(nn.Module):
    def __init__(self, modality, pretrain_avhubert_enc_video, pretrain_avhubert_enc_audio, 
                 pretrain_avhubert_enc_audiovisual, use_lora_avhubert, llm_model, hidden_size, intermediate_size, tokenizer, prompt, pad_id, 
                 downsample_ratio_audio, downsample_ratio_video, downsample_ratio_audiovisual, single_projector_avhubert, audio_encoder_name, 
                 unfrozen_modules, max_dec_tokens, num_beams, PETF_LLM_name = None, peft_config_llm = None,
                 use_uadf = False, uadf_fusion_method = 'uncertainty', uadf_temperature = 1.0,
                 ):
        
        super().__init__()
        
        self.modality = modality
        self.pretrain_avhubert_enc_video = pretrain_avhubert_enc_video
        self.pretrain_avhubert_enc_audio = pretrain_avhubert_enc_audio
        self.pretrain_avhubert_enc_audiovisual = pretrain_avhubert_enc_audiovisual
        self.max_dec_tokens = max_dec_tokens
        self.num_beams = num_beams
        self.downsample_ratio_audio = downsample_ratio_audio
        self.downsample_ratio_video = downsample_ratio_video
        self.downsample_ratio_audiovisual = downsample_ratio_audiovisual if modality == "audiovisual_avhubert" else None
        self.audio_encoder_name = audio_encoder_name
        self.llm_model = llm_model
        self.peft_config_llm = peft_config_llm
        self.PETF_LLM_name = PETF_LLM_name
        self.single_projector_avhubert = single_projector_avhubert
        self.use_uadf = use_uadf and modality == "audiovisual"  # UADF는 audiovisual 모달리티에서만 사용
        self.uadf_fusion_method = uadf_fusion_method
        self.uadf_temperature = uadf_temperature
            
        if modality == "audio" or modality == "audiovisual":
            
            if "whisper" in self.audio_encoder_name:
                print("Instantiating whisper!")    
                # 메모리 최적화: FP16으로 로드
                whisper_load_kwargs = {}
                if torch.cuda.is_available():
                    whisper_load_kwargs["torch_dtype"] = torch.float16
                self.audio_encoder = WhisperModel.from_pretrained(self.audio_encoder_name, **whisper_load_kwargs).encoder
                self.audio_frontend = AutoFeatureExtractor.from_pretrained(self.audio_encoder_name)
                self.audio_encoder.requires_grad_(False)
                self.audio_encoder.train() # This must be explicitly done as by default the from_pretrained HF models are in eval mode when initialized (this is the opposite for pytorch!)--> cause a break in deepspeed 3! https://github.com/Lightning-AI/pytorch-lightning/issues/19467
                audio_dim =self.audio_encoder.config.hidden_size
            
            elif "av-hubert" in self.audio_encoder_name:
                print("Initializing AV-HuBERT Large, non fine-tuned for ASR! Only audio goes through AV-HuBERT")
                
                modell, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([self.pretrain_avhubert_enc_audio]) #"/cappellazzo/AV_ASR/autoavsr_v1.1/results/large_vox_iter5.pt"
                self.audio_encoder = modell[0]
                self.audio_encoder.requires_grad_(False)
                audio_dim = 1024
            
            else: # WavLM.
                print("Instantiating WavLM!")    
                assert "wavlm" in self.audio_encoder_name, ("Only whisper and WavLM audio encoders are supported as of now!")
                self.audio_encoder = WavLMModel.from_pretrained(self.audio_encoder_name)
                self.audio_encoder.requires_grad_(False)
                audio_dim =self.audio_encoder.config.hidden_size
                
            # The projector is a two-layer MLP.
            self.audio_proj = nn.Sequential(nn.Linear(audio_dim*self.downsample_ratio_audio, intermediate_size), nn.ReLU(), nn.Linear(intermediate_size, hidden_size))
            
        if modality == "video" or modality == "audiovisual":
            assert pretrain_avhubert_enc_video is not None, ("The AV-HuBERT pre-trained model must be defined!")
            print("Initializing AV-HuBERT Large, non fine-tuned!")
            
            if use_lora_avhubert:
                
                modell, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([self.pretrain_avhubert_enc_video])
                self.video_encoder = modell[0]
                
                print('Preparing LoRA layers for AV-HuBERT video-only!')
                for layer_idx in range(24):
                    # We set apply_lora = True for each video encoder layer such that it is applied. TODO: define this parameter in the AV-HuBERT main class.
                    self.video_encoder.encoder.layers[layer_idx].apply_lora = True
                    
                    self.video_encoder.encoder.layers[layer_idx].self_attn.rank = 16
                    self.video_encoder.encoder.layers[layer_idx].self_attn.scaling_lora = 2
                    
                    self.video_encoder.encoder.layers[layer_idx].self_attn.lora_down_Q = nn.Linear(1024, round(1024/16), bias= False)
                    self.video_encoder.encoder.layers[layer_idx].self_attn.lora_up_Q = nn.Linear(round(1024/16), 1024, bias= False)
                    self.video_encoder.encoder.layers[layer_idx].self_attn.lora_down_V = nn.Linear(1024, round(1024/16), bias= False)
                    self.video_encoder.encoder.layers[layer_idx].self_attn.lora_up_V = nn.Linear(round(1024/16), 1024, bias= False)
        
                    nn.init.zeros_(self.video_encoder.encoder.layers[layer_idx].self_attn.lora_down_Q.weight)
                    nn.init.zeros_(self.video_encoder.encoder.layers[layer_idx].self_attn.lora_down_V.weight)
                    nn.init.kaiming_uniform_(self.video_encoder.encoder.layers[layer_idx].self_attn.lora_up_Q.weight, a=math.sqrt(5))
                    nn.init.kaiming_uniform_(self.video_encoder.encoder.layers[layer_idx].self_attn.lora_up_V.weight, a=math.sqrt(5))
            
            else:
                modell, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([self.pretrain_avhubert_enc_video])
                self.video_encoder = modell[0]
                
                
            self.video_encoder.requires_grad_(False)
            video_dim = 1024
             
            
            self.video_proj = nn.Sequential(nn.Linear(video_dim*self.downsample_ratio_video, intermediate_size), nn.ReLU(), nn.Linear(intermediate_size, hidden_size))
            
        
        if modality == "audiovisual_avhubert":
             print("Instantiating AV-HuBERT for audio-visual!")
             if use_lora_avhubert:
                 
                 modell, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([self.pretrain_avhubert_enc_audiovisual])
                 self.audiovisual_encoder = modell[0]
                 
                 print('Preparing LoRA layers for AV-HuBERT audio-visual!')
                 for layer_idx in range(24):
                     # We set apply_lora = True for each video encoder layer such that it is applied. TODO: define this parameter in the AV-HuBERT main class.
                     self.audiovisual_encoder.encoder.layers[layer_idx].apply_lora = True
                     
                     self.audiovisual_encoder.encoder.layers[layer_idx].self_attn.rank = 16
                     self.audiovisual_encoder.encoder.layers[layer_idx].self_attn.scaling_lora = 2
                     
                     self.audiovisual_encoder.encoder.layers[layer_idx].self_attn.lora_down_Q = nn.Linear(1024, round(1024/16), bias= False)
                     self.audiovisual_encoder.encoder.layers[layer_idx].self_attn.lora_up_Q = nn.Linear(round(1024/16), 1024, bias= False)
                     self.audiovisual_encoder.encoder.layers[layer_idx].self_attn.lora_down_V = nn.Linear(1024, round(1024/16), bias= False)
                     self.audiovisual_encoder.encoder.layers[layer_idx].self_attn.lora_up_V = nn.Linear(round(1024/16), 1024, bias= False)
         
                     nn.init.zeros_(self.audiovisual_encoder.encoder.layers[layer_idx].self_attn.lora_down_Q.weight)
                     nn.init.zeros_(self.audiovisual_encoder.encoder.layers[layer_idx].self_attn.lora_down_V.weight)
                     nn.init.kaiming_uniform_(self.audiovisual_encoder.encoder.layers[layer_idx].self_attn.lora_up_Q.weight, a=math.sqrt(5))
                     nn.init.kaiming_uniform_(self.audiovisual_encoder.encoder.layers[layer_idx].self_attn.lora_up_V.weight, a=math.sqrt(5))
             else:
                 modell, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([self.pretrain_avhubert_enc_audiovisual])
                 self.audiovisual_encoder = modell[0]
            
             self.audiovisual_encoder.requires_grad_(False)
             audiovisual_dim = 1024     
             
             if self.single_projector_avhubert:
                 self.audiovisual_proj = nn.Sequential(nn.Linear(audiovisual_dim*self.downsample_ratio_audiovisual, intermediate_size), nn.ReLU(), nn.Linear(intermediate_size, hidden_size))
             else:
                 self.audio_proj = nn.Sequential(nn.Linear(audiovisual_dim*self.downsample_ratio_audio, intermediate_size), nn.ReLU(), nn.Linear(intermediate_size, hidden_size))
                 self.video_proj = nn.Sequential(nn.Linear(audiovisual_dim*self.downsample_ratio_video, intermediate_size), nn.ReLU(), nn.Linear(intermediate_size, hidden_size))
                     
        
        # 메모리 최적화를 위한 로딩 옵션
        load_kwargs = {
            "low_cpu_mem_usage": True,  # CPU 메모리 사용량 최소화
        }
        
        # GPU가 있으면 FP16 사용 (메모리 절약)
        if torch.cuda.is_available():
            load_kwargs["torch_dtype"] = torch.float16
        
        if self.PETF_LLM_name is None:
            print(f"Loading LLM with memory optimization: {load_kwargs}")
            self.llm = LlamaForCausalLM.from_pretrained(llm_model, **load_kwargs)
        elif self.PETF_LLM_name == "lora":
            print(f"Loading LLM with LoRA...")
            # LoRA의 경우 from_pretrained가 다를 수 있으므로 기본 방식 사용
            self.llm = LlamaForCausalLM_lora.from_pretrained(llm_model, peft_config_llm)
            # 로드 후 FP16 변환 (메모리 절약)
            if torch.cuda.is_available():
                print("Converting LoRA model to FP16 for memory optimization...")
                self.llm = self.llm.half()
        else:
            raise Exception("Only LoRA is supported as PEFT method.")
                
        # IMPORTANT: we need to add the pad_id to the model and resize the token embeddings matrix accordingly.
        self.tokenizer = tokenizer
        self.llm.config.pad_token_id = pad_id
        
        self.llm.resize_token_embeddings(len(self.tokenizer))
        self.llm.requires_grad_(False)
        
        self.prompt = prompt
        
        # UADF 블록 초기화 (audiovisual 모달리티이고 UADF를 사용하는 경우)
        self.uadf_block = None
        if self.use_uadf:
            self.uadf_block = UADFBlock(
                hidden_size=hidden_size,
                fusion_method=uadf_fusion_method,
                temperature=uadf_temperature
            )
            print(f"✅ UADF 블록 초기화 완료 (fusion_method={uadf_fusion_method}, temperature={uadf_temperature})")
        
        self._unfreeze_PETF(unfrozen_modules)
        
        
    def _unfreeze_PETF(self, unfrozen_modules):
        """
        Modules to be unfrozen. Unfrozen_blocks is a list with one or multiple values.
        """
        if None in unfrozen_modules:
            return
        if "peft_llm" in unfrozen_modules:
            print("Unfreezing LoRa for LLM:")
            for block_idx in range(self.llm.config.num_hidden_layers):
                self.llm.model.layers[block_idx].self_attn.lora_down_Q.requires_grad_(True)
                self.llm.model.layers[block_idx].self_attn.lora_up_Q.requires_grad_(True)
                self.llm.model.layers[block_idx].self_attn.lora_down_V.requires_grad_(True)
                self.llm.model.layers[block_idx].self_attn.lora_up_V.requires_grad_(True)
        if "lora_avhubert" in unfrozen_modules:
            
            if self.modality == "video": 
                print("Unfreezing LoRA for AV-HuBERT video encoder!")
                
                for block_idx in range(24):
                    self.video_encoder.encoder.layers[block_idx].self_attn.lora_down_Q.requires_grad_(True)
                    self.video_encoder.encoder.layers[block_idx].self_attn.lora_up_Q.requires_grad_(True)
                    self.video_encoder.encoder.layers[block_idx].self_attn.lora_down_V.requires_grad_(True)
                    self.video_encoder.encoder.layers[block_idx].self_attn.lora_up_V.requires_grad_(True)
                
            else:
                assert self.modality == "audiovisual_avhubert"
                print("Unfreezing LoRA for AV-HuBERT audiovisual encoder!")
                for block_idx in range(24):
                    self.audiovisual_encoder.encoder.layers[block_idx].self_attn.lora_down_Q.requires_grad_(True)
                    self.audiovisual_encoder.encoder.layers[block_idx].self_attn.lora_up_Q.requires_grad_(True)
                    self.audiovisual_encoder.encoder.layers[block_idx].self_attn.lora_down_V.requires_grad_(True)
                    self.audiovisual_encoder.encoder.layers[block_idx].self_attn.lora_up_V.requires_grad_(True)
    
    def forward(self, inputs, is_trainval= True):
        
        text_embeddings, labels = self.prepare_inputs(inputs, is_trainval)
        
        if is_trainval: # Train/eval step: compute the logits and loss. Note that the llm computes itself the loss since we feed labels.
            outputs = self.llm(inputs_embeds = text_embeddings, labels = labels)
            
            return outputs
        
        else: # Inference step: we decode starting from the audio/video tokens + bos. 
            
            # [수정] 모델 이름에 "Llama-3"가 포함되어 있으면 Llama-3 설정 사용 (로컬 경로 호환)
            if "Llama-3" in self.llm_model:
                decoded_ids = self.llm.generate(inputs_embeds = text_embeddings, max_new_tokens = self.max_dec_tokens, num_beams=self.num_beams, eos_token_id = self.tokenizer.vocab["<|end_of_text|>"], 
                                                bos_token_id = self.tokenizer.vocab["<|begin_of_text|>"], 
                                                pad_token_id = self.tokenizer.vocab["<pad>"],
                                                )
            # Llama-2 계열 (TinyLlama 등)
            else: 
                decoded_ids = self.llm.generate(inputs_embeds = text_embeddings, max_new_tokens = self.max_dec_tokens, num_beams=self.num_beams, eos_token_id = self.tokenizer.vocab["</s>"], 
                                                bos_token_id = self.tokenizer.vocab["<s>"], 
                                                pad_token_id = self.tokenizer.vocab["<pad>"],
                                                )
                
            return decoded_ids
            
    
    # We follow Macaw LLM (https://github.com/lyuchenyang/Macaw-LLM) and we use BOS/EOS tokens both for audio and video as delimiters. 
    def prepare_inputs(self, inputs, is_trainval):
        
        # AV-HuBERT processes both audio and video tokens. In the paper, we show that this approach is suboptimal compared to
        # using AV-HuBERT for video and Whisper for audio.
        if self.modality == "audiovisual_avhubert":
            
            # In this case, a single projector processes the audio-visual output of AV-HuBERT. Otherwise, we compute the audio and video tokens
            # by computing twice AV-HuBERT and setting the other modality to None and we obtain independent audio and video tokens which go through separate projectors .
            # From our experiments, both approaches lead to very similar results.
            if self.single_projector_avhubert:
            
                audiovisual_features = self.encode_AVH_audiovisual(inputs["audio"], inputs["video"], self.single_projector_avhubert)
                
                text_embeddings_ = self.llm.model.embed_tokens(inputs["tokens"])
                ignore_count = 0 
                
                prompt_ids = self.tokenizer(self.prompt, return_tensors = "pt").input_ids[:,1:-1].to(text_embeddings_.device)
                prompt_embeddings = self.llm.model.embed_tokens(prompt_ids.expand(inputs["tokens"].shape[0],-1))
                
                if is_trainval:
                    text_embeddings = torch.cat(
                        [torch.cat([text_embeddings_[:, 0, :].unsqueeze(1), prompt_embeddings], dim=1), text_embeddings_[:, 1:, :]], 
                        dim=1)
                else:
                    text_embeddings = torch.cat([text_embeddings_[:, 0, :].unsqueeze(1), prompt_embeddings], dim=1)
                
                ignore_count += prompt_embeddings.shape[1]
                
                audiovisual_features = self.audiovisual_proj(audiovisual_features)
                
                text_embeddings = torch.cat(
                    [torch.cat([text_embeddings[:, 0, :].unsqueeze(1), audiovisual_features], dim=1), text_embeddings[:, 1:, :]], 
                    dim=1)
                ignore_count += audiovisual_features.shape[1]
                
                if inputs["labels"] is not None:
                    labels = torch.tensor([IGNORE_INDEX]*ignore_count, device=text_embeddings.device).expand(text_embeddings.shape[0], -1)
                    labels = torch.cat(
                        [torch.cat([inputs["labels"][:, 0].unsqueeze(1), labels], dim=1), inputs["labels"][:, 1:]], 
                        dim=1)
                else:
                    labels = None
                
                return text_embeddings, labels
            else:
                audio_features, video_features = self.encode_AVH_audiovisual(inputs["audio"], inputs["video"], self.single_projector_avhubert)
                
                text_embeddings_ = self.llm.model.embed_tokens(inputs["tokens"])
                ignore_count = 0 
                
                prompt_ids = self.tokenizer(self.prompt, return_tensors = "pt").input_ids[:,1:-1].to(text_embeddings_.device)
                prompt_embeddings = self.llm.model.embed_tokens(prompt_ids.expand(inputs["tokens"].shape[0],-1))
                
                if is_trainval:
                    text_embeddings = torch.cat(
                        [torch.cat([text_embeddings_[:, 0, :].unsqueeze(1), prompt_embeddings], dim=1), text_embeddings_[:, 1:, :]], 
                        dim=1)
                else:
                    text_embeddings = torch.cat([text_embeddings_[:, 0, :].unsqueeze(1), prompt_embeddings], dim=1)
                
                ignore_count += prompt_embeddings.shape[1]
                
                if video_features is not None:
                    video_starts = torch.tensor([self.tokenizer.vocab["<video>"]], device = text_embeddings.device).expand(inputs["tokens"].shape[0],-1)
                    video_starts = self.llm.model.embed_tokens(video_starts)
                    
                    video_ends = torch.tensor([self.tokenizer.vocab["</video>"]], device = text_embeddings.device).expand(inputs["tokens"].shape[0],-1)
                    video_ends = self.llm.model.embed_tokens(video_ends)
                    
                    video_features = self.video_proj(video_features)
                    
                    video_inputs = torch.cat([torch.cat([video_starts, video_features], dim=1), video_ends], dim=1)
                    
                    text_embeddings = torch.cat(
                        [torch.cat([text_embeddings[:, 0, :].unsqueeze(1), video_inputs], dim=1), text_embeddings[:, 1:, :]], 
                        dim=1)
                    ignore_count += video_inputs.shape[1]
                
                if audio_features is not None:
                    audio_starts = torch.tensor([self.tokenizer.vocab["<audio>"]], device = text_embeddings.device).expand(inputs["tokens"].shape[0],-1)
                    audio_starts =  self.llm.model.embed_tokens(audio_starts)
                    
                    audio_ends = torch.tensor([self.tokenizer.vocab["</audio>"]], device = text_embeddings.device).expand(inputs["tokens"].shape[0],-1)
                    audio_ends = self.llm.model.embed_tokens(audio_ends)
                    
                    audio_features = self.audio_proj(audio_features)
                    
                    audio_inputs = torch.cat([torch.cat([audio_starts, audio_features], dim=1), audio_ends], dim=1)
                    
                    text_embeddings = torch.cat(
                        [torch.cat([text_embeddings[:, 0, :].unsqueeze(1), audio_inputs], dim=1), text_embeddings[:, 1:, :]], 
                        dim=1)
                    ignore_count += audio_inputs.shape[1]
            
                if inputs["labels"] is not None:
                    labels = torch.tensor([IGNORE_INDEX]*ignore_count, device=text_embeddings.device).expand(text_embeddings.shape[0], -1)
                    labels = torch.cat(
                        [torch.cat([inputs["labels"][:, 0].unsqueeze(1), labels], dim=1), inputs["labels"][:, 1:]], 
                        dim=1)
                else:
                    labels = None
                
                return text_embeddings, labels
                
        else:
        
            audio_features = self.encode_audio(inputs["audio"], max(inputs["lengths"]), is_trainval) if self.modality in ["audio", "audiovisual"] else None
            video_features = self.encode_video(inputs["video"]) if self.modality in ["video", "audiovisual"] else None
            
            
            text_embeddings_ = self.llm.model.embed_tokens(inputs["tokens"])
            
            ignore_count = 0 
            
            
            # An important note here: the tokenizer by default inserts the EOS and BOS tokens. Since we do that already in the collate_LLM, here we need to
            # get rid of them explicitly --> [:,1:-1].
            prompt_ids = self.tokenizer(self.prompt, return_tensors = "pt").input_ids[:,1:-1].to(text_embeddings_.device)
            prompt_embeddings = self.llm.model.embed_tokens(prompt_ids.expand(inputs["tokens"].shape[0],-1))
            
            if is_trainval:
                text_embeddings = torch.cat(
                    [torch.cat([text_embeddings_[:, 0, :].unsqueeze(1), prompt_embeddings], dim=1), text_embeddings_[:, 1:, :]], 
                    dim=1)
            else:
                text_embeddings = torch.cat([text_embeddings_[:, 0, :].unsqueeze(1), prompt_embeddings], dim=1)
            
            ignore_count += prompt_embeddings.shape[1]
            
            # UADF를 사용하는 경우: 오디오와 비디오 피처를 융합
            if self.use_uadf and video_features is not None and audio_features is not None:
                # 프로젝션
                video_features_proj = self.video_proj(video_features)
                audio_features_proj = self.audio_proj(audio_features)
                
                # UADF로 융합
                fused_features = self.uadf_block(audio_features_proj, video_features_proj)
                
                # 융합된 피처를 LLM에 전달
                audiovisual_starts = torch.tensor([self.tokenizer.vocab["<video>"]], device = text_embeddings.device).expand(inputs["tokens"].shape[0],-1)
                audiovisual_starts = self.llm.model.embed_tokens(audiovisual_starts)
                
                audiovisual_ends = torch.tensor([self.tokenizer.vocab["</video>"]], device = text_embeddings.device).expand(inputs["tokens"].shape[0],-1)
                audiovisual_ends = self.llm.model.embed_tokens(audiovisual_ends)
                
                audiovisual_inputs = torch.cat([torch.cat([audiovisual_starts, fused_features], dim=1), audiovisual_ends], dim=1)
                
                text_embeddings = torch.cat(
                    [torch.cat([text_embeddings[:, 0, :].unsqueeze(1), audiovisual_inputs], dim=1), text_embeddings[:, 1:, :]], 
                    dim=1)
                ignore_count += audiovisual_inputs.shape[1]
            
            else:
                # 기존 방식: 오디오와 비디오를 각각 처리
                if video_features is not None:
                    video_starts = torch.tensor([self.tokenizer.vocab["<video>"]], device = text_embeddings.device).expand(inputs["tokens"].shape[0],-1)
                    video_starts = self.llm.model.embed_tokens(video_starts)
                    
                    video_ends = torch.tensor([self.tokenizer.vocab["</video>"]], device = text_embeddings.device).expand(inputs["tokens"].shape[0],-1)
                    video_ends = self.llm.model.embed_tokens(video_ends)
                    
                    video_features = self.video_proj(video_features)
                    
                    video_inputs = torch.cat([torch.cat([video_starts, video_features], dim=1), video_ends], dim=1)
                    
                    text_embeddings = torch.cat(
                        [torch.cat([text_embeddings[:, 0, :].unsqueeze(1), video_inputs], dim=1), text_embeddings[:, 1:, :]], 
                        dim=1)
                    ignore_count += video_inputs.shape[1]
                
                if audio_features is not None:
                    audio_starts = torch.tensor([self.tokenizer.vocab["<audio>"]], device = text_embeddings.device).expand(inputs["tokens"].shape[0],-1)
                    audio_starts = self.llm.model.embed_tokens(audio_starts)
                    
                    audio_ends = torch.tensor([self.tokenizer.vocab["</audio>"]], device = text_embeddings.device).expand(inputs["tokens"].shape[0],-1)
                    audio_ends = self.llm.model.embed_tokens(audio_ends)
                    
                    audio_features = self.audio_proj(audio_features)
                    audio_inputs = torch.cat([torch.cat([audio_starts, audio_features], dim=1), audio_ends], dim=1)
                    
                    text_embeddings = torch.cat(
                        [torch.cat([text_embeddings[:, 0, :].unsqueeze(1), audio_inputs], dim=1), text_embeddings[:, 1:, :]], 
                        dim=1)
                    ignore_count += audio_inputs.shape[1]
                
            if inputs["labels"] is not None:
                labels = torch.tensor([IGNORE_INDEX]*ignore_count, device=text_embeddings.device).expand(text_embeddings.shape[0], -1)
                labels = torch.cat(
                    [torch.cat([inputs["labels"][:, 0].unsqueeze(1), labels], dim=1), inputs["labels"][:, 1:]], 
                    dim=1)
            else:
                labels = None
            
            return text_embeddings, labels
    
    
    def encode_AVH_audiovisual(self, audios, videos, single_projector_avhubert):
        if single_projector_avhubert:
            audiovisual_temp = self.audiovisual_encoder.extract_finetune(source={'video': torch.reshape(videos,(-1,videos.shape[2],videos.shape[1],videos.shape[3],videos.shape[-1])),'audio': audios.transpose(1, 2)})[0]
            if self.downsample_ratio_audiovisual != 1:
                audiovisual_enc = [audiovisual_temp[:, x:x + self.downsample_ratio_audiovisual, :].view(audiovisual_temp.shape[0], 1, -1) for x in range(0, audiovisual_temp.shape[1], self.downsample_ratio_audiovisual)]
                rest = audiovisual_temp.shape[1] % self.downsample_ratio_audiovisual
                if rest == 0:
                    audiovisual_enc = torch.stack(audiovisual_enc, dim=1).squeeze(2)
                else:
                    audiovisual_enc = torch.stack(audiovisual_enc[:-1], dim=1).squeeze(2)
            return audiovisual_enc
        else:
            
            video_enc = self.audiovisual_encoder.extract_finetune(source={'video': torch.reshape(videos,(-1,videos.shape[2],videos.shape[1],videos.shape[3],videos.shape[-1])),'audio': None})[0]
            if self.downsample_ratio_video != 1:
                video_enc = [video_enc[:, x:x + self.downsample_ratio_video, :].view(video_enc.shape[0], 1, -1) for x in range(0, video_enc.shape[1], self.downsample_ratio_video)]
                video_enc = torch.stack(video_enc, dim=1).squeeze(2)
            
            audio_temp = self.audiovisual_encoder.extract_finetune(source={'audio': audios.transpose(1, 2), 'video': None})[0]
            
            if self.downsample_ratio_audio != 1:
                audio_enc = [audio_temp[:, x:x + self.downsample_ratio_audio, :].view(audio_temp.shape[0], 1, -1) for x in range(0, audio_temp.shape[1], self.downsample_ratio_audio)]
                rest = audio_temp.shape[1] % self.downsample_ratio_audio
                if rest == 0:
                    audio_enc = torch.stack(audio_enc, dim=1).squeeze(2)
                else:
                    audio_enc = torch.stack(audio_enc[:-1], dim=1).squeeze(2)
            
            return audio_enc, video_enc
    
    def encode_video(self, videos):
            
        video_enc = self.video_encoder.extract_finetune(source={'video': torch.reshape(videos,(-1,videos.shape[2],videos.shape[1],videos.shape[3],videos.shape[-1])),'audio': None})[0]
        if self.downsample_ratio_video != 1:
            video_enc = [video_enc[:, x:x + self.downsample_ratio_video, :].view(video_enc.shape[0], 1, -1) for x in range(0, video_enc.shape[1], self.downsample_ratio_video)]
            video_enc = torch.stack(video_enc, dim=1).squeeze(2)
            
        return video_enc
    
    def encode_audio(self, audios, max_len, is_trainval):
            
        if "whisper" in self.audio_encoder_name:
            audios = audios.to(torch.float32)
            
            audios = audios.cpu().numpy()
            audio_extract = self.audio_frontend(audios.squeeze(-1), return_tensors="pt",sampling_rate =16000).input_features

            # ...
            # [해결] .half() 대신 모델의 dtype을 따라가도록 변경
            target_dtype = self.audio_encoder.dtype
            audio_enc = self.audio_encoder(audio_extract.cuda().to(target_dtype)).last_hidden_state


            # Due to the 30s padding required by Whisper, we drop the tokens that correspond to the padded 0s. As 1s corresponds to 50 tokens, we truncate acccordingly.
            audio_enc = audio_enc[:, 0: max(int(max_len/16000*50), 25) , :]
            
            if self.downsample_ratio_audio != 1:
                audio_temp = audio_enc
                audio_enc = [audio_temp[:, x:x + self.downsample_ratio_audio, :].view(audio_temp.shape[0], 1, -1) for x in range(0, audio_temp.shape[1], self.downsample_ratio_audio)]
                rest = audio_temp.shape[1] % self.downsample_ratio_audio
                if rest == 0:
                    audio_enc = torch.stack(audio_enc, dim=1).squeeze(2) 
                else: 
                    audio_enc = torch.stack(audio_enc[:-1], dim=1).squeeze(2)
                
        # We also tested the case where we AV-HuBERT to process the audio.
        elif "av-hubert" in self.audio_encoder_name:
            audio_temp = self.audio_encoder.extract_finetune(source={'audio': audios.transpose(1, 2), 'video': None})[0]
            
            if self.downsample_ratio_audio != 1:
                audio_enc = [audio_temp[:, x:x + self.downsample_ratio_audio, :].view(audio_temp.shape[0], 1, -1) for x in range(0, audio_temp.shape[1], self.downsample_ratio_audio)]
                rest = audio_temp.shape[1] % self.downsample_ratio_audio
                if rest == 0:
                    audio_enc = torch.stack(audio_enc, dim=1).squeeze(2)
                else:
                    audio_enc = torch.stack(audio_enc[:-1], dim=1).squeeze(2)
        
        #WavLM audio encoder.
        else:
            audio_temp = self.audio_encoder(audios.squeeze(-1))[0]
            
            if self.downsample_ratio_audio != 1:
                audio_enc = [audio_temp[:, x:x + self.downsample_ratio_audio, :].view(audio_temp.shape[0], 1, -1) for x in range(0, audio_temp.shape[1], self.downsample_ratio_audio)]
                rest = audio_temp.shape[1] % self.downsample_ratio_audio
                if rest == 0:
                    audio_enc = torch.stack(audio_enc, dim=1).squeeze(2)
                else:
                    audio_enc = torch.stack(audio_enc[:-1], dim=1).squeeze(2)
            else:
                audio_enc = audio_temp
        
        return audio_enc