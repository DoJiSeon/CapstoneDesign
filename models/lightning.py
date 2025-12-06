import sys
sys.path.append("..")
import torch
import torchaudio
from utils.cosine import WarmupCosineScheduler
from pytorch_lightning import LightningModule
from transformers import AutoTokenizer
from .Llama_LoRA import LoRA_config
from .modeling_AVSRLLM import AVSR_LLMs
from tokenizers.processors import TemplateProcessing

DEFAULT_PAD_TOKEN = "<pad>"
AUDIO_SOS = "<audio>"
AUDIO_EOS = "</audio>"
VIDEO_SOS = "<video>"
VIDEO_EOS = "</video>"

llm_size = {"TinyLlama/TinyLlama_v1.1": 2048,
            "meta-llama/Llama-2-13b-hf": 5120,
            "meta-llama/Llama-2-7b-hf": 4096,
            "meta-llama/Meta-Llama-3.1-8B": 4096,
            # 로컬 경로 패턴 (일반적인 경로들)
            "models/Meta-Llama-3.1-8B": 4096,
            "models/TinyLlama_v1.1": 2048,
            }

def get_llm_hidden_size(llm_model_path):
    """LLM 모델 경로에서 hidden_size를 가져옵니다."""
    # 정확한 경로 매칭 시도
    if llm_model_path in llm_size:
        return llm_size[llm_model_path]
    
    # 경로에서 모델 이름 추출하여 매칭 시도
    llm_model_lower = llm_model_path.lower()
    if "tinyllama" in llm_model_lower:
        return 2048
    elif "llama-2-13b" in llm_model_lower or "llama-2-13b-hf" in llm_model_lower:
        return 5120
    elif "llama-2-7b" in llm_model_lower or "llama-2-7b-hf" in llm_model_lower:
        return 4096
    elif "llama-3.1-8b" in llm_model_lower or "meta-llama-3.1-8b" in llm_model_lower:
        return 4096
    elif "llama-3" in llm_model_lower:
        return 4096  # 기본값
    
    # 기본값 반환 (TinyLlama 크기)
    print(f"⚠️  경고: '{llm_model_path}'에 대한 hidden_size를 찾을 수 없어 기본값 2048을 사용합니다.")
    return 2048


def compute_word_level_distance(seq1, seq2):
    seq1, seq2 = seq1.lower().split(), seq2.lower().split()
    return torchaudio.functional.edit_distance(seq1, seq2)

class ModelModule_LLM(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)
        
        if args.use_lora_avhubert:
            assert "lora_avhubert" in args.unfrozen_modules, ("LoRA modules for the AV-HuBERT encoder must be unfrozen!!")
        
        self.tokenizer = AutoTokenizer.from_pretrained(args.llm_model, add_bos_token=True, add_eos_token= True)
        
        # Apparently, some LLMs don't rely on FastTokenizer and it seems like they don't append the EOS token even though you set
        # it explicitly. In my case, this happens for LLama3. More details at: https://github.com/huggingface/transformers/issues/22794.
        
        # Llama3 모델 체크 (경로도 고려)
        is_llama3 = (args.llm_model == "meta-llama/Meta-Llama-3.1-8B" or 
                     "Meta-Llama-3.1-8B" in args.llm_model or 
                     "llama-3.1-8b" in args.llm_model.lower())
        if is_llama3:
            bos = self.tokenizer.bos_token
            eos = self.tokenizer.eos_token
            
            self.tokenizer._tokenizer.post_processor =TemplateProcessing(
                single=f"{bos}:0 $A:0 {eos}:0",
                pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
                special_tokens=[
                    (f"{bos}", self.tokenizer.bos_token_id), 
                    (f"{eos}", self.tokenizer.eos_token_id)
                    ],
                )
        
        # By default, LLaMA doesn't come with a padding token (pad_token= None), so we need to introduce it.
        num_added_toks = self.tokenizer.add_special_tokens({"pad_token": DEFAULT_PAD_TOKEN, "additional_special_tokens": [AUDIO_SOS, AUDIO_EOS, VIDEO_SOS, VIDEO_EOS]})
        pad_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_PAD_TOKEN)
       
            
        print("We have added ", num_added_toks, " tokens to the tokenizer!")
        self.tokenizer.padding_side = "right"   # The padding is added to the right.
        
        # The resize of the embed_tokens matrix and the add of the pad_token to the model is performed when the model is called.
        
        if args.modality == 'audio':
            prompt = args.prompt_audio
        elif args.modality == 'video':
            prompt = args.prompt_video
        else:
            assert args.modality in ['audiovisual', 'audiovisual_avhubert']
            prompt = args.prompt_audiovisual
        
        print(f"The prompt used for the {args.modality} modality is: {prompt}")
        
        if args.add_PETF_LLM:
            
            # Llama3 모델 체크 (경로도 고려)
            IS_LLAMA3 = (args.llm_model == "meta-llama/Meta-Llama-3.1-8B" or 
                         "Meta-Llama-3.1-8B" in args.llm_model or 
                         "llama-3.1-8b" in args.llm_model.lower())
            IS_TINYLLAMA = True if "TinyLlama" in args.llm_model or "tinyllama" in args.llm_model.lower() else False
            lora_config_llm = LoRA_config(args.reduction_lora, args.alpha, IS_LLAMA3, IS_TINYLLAMA)
            
            self.model = AVSR_LLMs(modality = args.modality,  
                                   pretrain_avhubert_enc_video = args.pretrain_avhubert_enc_video_path, 
                                   pretrain_avhubert_enc_audio = args.pretrain_avhubert_enc_audio_path, 
                                   pretrain_avhubert_enc_audiovisual = args.pretrain_avhubert_enc_audiovisual_path,
                                   use_lora_avhubert= args.use_lora_avhubert,
                                   llm_model = args.llm_model, 
                                   hidden_size = get_llm_hidden_size(args.llm_model), 
                                   intermediate_size= args.intermediate_size, 
                                   tokenizer = self.tokenizer, 
                                   prompt = prompt, 
                                   pad_id = pad_id, 
                                   downsample_ratio_audio = args.downsample_ratio_audio, 
                                   downsample_ratio_video = args.downsample_ratio_video, 
                                   downsample_ratio_audiovisual = args.downsample_ratio_audiovisual,
                                   single_projector_avhubert = args.single_projector_avhubert,
                                   audio_encoder_name = args.audio_encoder_name,
                                   unfrozen_modules= args.unfrozen_modules, 
                                   max_dec_tokens = args.max_dec_tokens, 
                                   num_beams = args.num_beams, 
                                   PETF_LLM_name = args.add_PETF_LLM, 
                                   peft_config_llm= lora_config_llm,
                                   use_uadf = getattr(args, 'use_uadf', False),
                                   uadf_fusion_method = getattr(args, 'uadf_fusion_method', 'uncertainty'),
                                   uadf_temperature = getattr(args, 'uadf_temperature', 1.0),
                                   )
            
            n_parameters_learn = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print("Total number of trainable parameters of the model: ", n_parameters_learn)
        
        else:
            self.model = AVSR_LLMs(modality = args.modality, 
                                   pretrain_avhubert_enc_video = args.pretrain_avhubert_enc_video_path,
                                   pretrain_avhubert_enc_audio = args.pretrain_avhubert_enc_audio_path, 
                                   pretrain_avhubert_enc_audiovisual = args.pretrain_avhubert_enc_audiovisual_path,
                                   use_lora_avhubert= args.use_lora_avhubert,
                                   llm_model = args.llm_model,
                                   hidden_size = get_llm_hidden_size(args.llm_model),
                                   intermediate_size= args.intermediate_size,
                                   tokenizer = self.tokenizer,
                                   prompt = prompt,
                                   pad_id = pad_id,
                                   downsample_ratio_audio = args.downsample_ratio_audio,
                                   downsample_ratio_video = args.downsample_ratio_video,
                                   downsample_ratio_audiovisual = args.downsample_ratio_audiovisual,
                                   single_projector_avhubert = args.single_projector_avhubert,
                                   audio_encoder_name = args.audio_encoder_name,
                                   unfrozen_modules= args.unfrozen_modules,
                                   max_dec_tokens = args.max_dec_tokens,
                                   num_beams = args.num_beams,
                                   use_uadf = getattr(args, 'use_uadf', False),
                                   uadf_fusion_method = getattr(args, 'uadf_fusion_method', 'uncertainty'),
                                   uadf_temperature = getattr(args, 'uadf_temperature', 1.0),
                                   )
            
            n_parameters_learn = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print("Total number of trainable parameters of the model: ", n_parameters_learn)

            
            
        
        # initialize the full model from the checkpoint for inference.
        if args.pretrained_model_path:
            ckpt = torch.load(args.pretrained_model_path)
            self.model.load_state_dict(ckpt, strict=False)
            
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params= self.model.parameters(), lr= self.args.lr, weight_decay=self.args.weight_decay, betas=(0.9, 0.98))
        scheduler = WarmupCosineScheduler(optimizer, self.args.warmup_epochs, self.args.max_epochs, len(self.trainer.datamodule.train_dataloader()) / self.trainer.num_devices / self.trainer.num_nodes)
        
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
        
    def training_step(self, batch, batch_idx):
        try:
            # 1. 모델 실행
            output = self.model(batch, is_trainval=True)
            
            # 2. [핵심 수정] Loss 값 안전하게 꺼내기
            # 원래 코드의 [0] 방식은 Llama 모델에서 안 먹힙니다.
            if hasattr(output, 'loss'):
                train_loss = output.loss  # 객체 안에 있는 .loss를 꺼냄
            elif isinstance(output, tuple):
                train_loss = output[0]    # 튜플이면 첫 번째꺼 꺼냄
            else:
                train_loss = output       # 그냥 텐서면 그대로 씀

            # 3. 원래 코드의 로직 유지 (배치 사이즈 계산 및 로깅)
            batch_size = batch["tokens"].shape[0]

            # 'loss'와 'train_loss' 둘 다 기록 (WandB 확인용)
            self.log("loss", train_loss, on_step=True, on_epoch=True, batch_size=batch_size, prog_bar=True)
            self.log("train_loss", train_loss, on_step=True, on_epoch=True, batch_size=batch_size)
            
            # 분산 학습(DDP)시 배치 크기 보정 로직 (원래 코드 유지)
            # (단, 싱글 GPU면 batch_sizes.size(0)은 1이 되므로 영향 없음)
            if self.trainer.num_devices > 1:
                batch_sizes = self.all_gather(batch_size)
                train_loss_scaled = train_loss * (batch_sizes.size(0) / batch_sizes.sum())
            else:
                train_loss_scaled = train_loss

            # 4. [중요] 모니터링 지표 저장
            # 원래 코드는 global_step을 저장했는데, 체크포인트 저장을 위해서는 'loss'가 더 유용합니다.
            self.log("monitoring_step", train_loss_scaled)
            
            return train_loss_scaled

        # 5. [필수] 짧은 비디오 에러 방지 (Skiping)
        except RuntimeError as e:
            if "Kernel size can't be greater than actual input size" in str(e):
                print(f"\n⚠️ [SKIP] 짧은 비디오 데이터 발견! 학습을 건너뜁니다. (Batch: {batch_idx})")
                return None 
            else:
                raise e
        
    def validation_step(self, batch, batch_idx):
        val_loss = self.model(batch, is_trainval = True)[0]
        
        batch_size = batch["tokens"].shape[0]
        
        self.log("loss_val", val_loss, batch_size=batch_size, sync_dist=True)
        
        return val_loss
            
    def test_step(self, batch, batch_idx):
        
        generated_ids = self.model(batch, is_trainval = False)
        generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        print("Input text: ", batch["gold_text"])
        print("Generated text: ", generated_text)
        
        self.total_edit_distance += compute_word_level_distance(batch["gold_text"], generated_text)
        self.total_length += len(batch["gold_text"].split())
        return
    
    def on_test_epoch_start(self):
        self.total_length = 0
        self.total_edit_distance = 0
        
    def on_test_epoch_end(self):
        self.log("wer", self.total_edit_distance / self.total_length)

    # [추가할 코드] 저장할 때 뚱뚱한 베이스 모델은 빼고, 학습된 LoRA만 남기기
    def on_save_checkpoint(self, checkpoint):
        # state_dict(모든 파라미터) 중에서 'lora', 'video_proj' 등이 들어간 것만 필터링
        # (modules_to_save는 Peft 라이브러리가 지정한 추가 저장 모듈들)
        keys_to_keep = [k for k in checkpoint['state_dict'].keys() 
                        if "lora" in k or "modules_to_save" in k or "video_proj" in k]
        
        new_state_dict = {k: checkpoint['state_dict'][k] for k in keys_to_keep}
        
        # 필터링된 가벼운 딕셔너리로 교체
        checkpoint['state_dict'] = new_state_dict