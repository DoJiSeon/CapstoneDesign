import logging
import os
from argparse import ArgumentParser, ArgumentTypeError
import torch
import time
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# [수정 1] 기존 모듈 임포트 아래에 GRIDDataset 임포트 추가
from utils.avg_checkpoints_original import ensemble_original
from datamodule.data_module import DataModule_LLM, collate_LLM
from datamodule.grid_dataset import GRIDDataset  # <--- 추가됨
from models.lightning import ModelModule_LLM

from pytorch_lightning import seed_everything, Trainer, LightningDataModule # <--- LightningDataModule 추가
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def str2bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value_lower = value.lower()
        if value_lower in {"true", "t", "1", "yes", "y"}:
            return True
        if value_lower in {"false", "f", "0", "no", "n"}:
            return False
    raise ArgumentTypeError(f"Boolean value expected, got '{value}'.")

# [수정 2] GRID 데이터셋을 위한 LightningDataModule 래퍼 클래스 정의
class GRIDDataModule(LightningDataModule):
    def __init__(self, args, tokenizer, train_num_buckets=None):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.batch_size = 1 # 기본 배치 1로 설정 (필요시 args.batch_size로 변경)
        self.num_workers = 4

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = GRIDDataset(
                root_dir=self.args.root_dir,
                modality=self.args.modality,
                split="train",
                split_ratio=self.args.grid_split_ratio,
                max_samples=self.args.grid_max_train_samples,
                shuffle=True
            )
            self.val_dataset = GRIDDataset(
                root_dir=self.args.root_dir,
                modality=self.args.modality,
                split="val",
                split_ratio=self.args.grid_split_ratio,
                max_samples=self.args.grid_max_val_samples,
                shuffle=False
            )
        
        if stage == 'test':
            self.test_dataset = GRIDDataset(
                root_dir=self.args.root_dir,
                modality=self.args.modality,
                split="test",
                split_ratio=self.args.grid_split_ratio,
                max_samples=self.args.grid_max_test_samples,
                shuffle=False
            )

    def train_dataloader(self):
        # collate_fn=self.collate_fn이 여기서 호출됩니다.
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                          shuffle=True, num_workers=self.num_workers, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, 
                          shuffle=False, num_workers=self.num_workers, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, 
                          shuffle=False, num_workers=self.num_workers, collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        # GRIDDataset이 반환하는 sample dict들 그대로 collate_LLM에 넘김
        # sample 안에 "audio", "video", "tokens"가 이미 들어있으니까 그대로 사용 가능
        return collate_LLM(
            batch,
            self.tokenizer,
            self.args.modality,
            is_trainval=True  # 학습/검증용
        )    


def get_trainer(args):
    seed_everything(args.seed, workers=True)
    checkpoint = ModelCheckpoint(
        dirpath=os.path.join(args.exp_dir, args.exp_name) if args.exp_dir else None,
        monitor="monitoring_step",
        mode="max",
        save_last=False,
        filename="{epoch}",
        save_top_k=1,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint, lr_monitor]

    find_unused_parameters_flag = False

    return Trainer(
        precision='bf16-true',
        sync_batchnorm=True,
        num_sanity_val_steps=2,
        default_root_dir=args.exp_dir,
        max_epochs=args.max_epochs,
        num_nodes=args.num_nodes,
        devices=args.gpus,
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=find_unused_parameters_flag),  
        callbacks=callbacks,
        reload_dataloaders_every_n_epochs=1,
        logger=WandbLogger(name=args.exp_name, project=args.project_wandb),
        gradient_clip_val=10.0,
        val_check_interval=args.val_check_interval,
        accumulate_grad_batches=args.accumulate_grad_batches,
    )

def get_test_trainer(args):
    return Trainer(precision='bf16-true',
        num_nodes=1,
        devices=1,
        accelerator="gpu",
        logger=WandbLogger(name=args.exp_name, project=args.project_wandb),
    )

def parse_args():
    parser = ArgumentParser()
    # ... (기존 인자들은 그대로 유지) ...
    parser.add_argument("--exp-dir", default=None, type=str)
    parser.add_argument("--root-dir", default=None, type=str, help="Root directory of dataset")
    parser.add_argument("--dataset-name", default="lrs3", type=str, choices=["lrs3", "grid"])
    
    # GRID 관련 인자 (기존 코드에 이미 포함되어 있음)
    parser.add_argument("--grid-split-ratio", nargs=3, type=float, default=[0.8, 0.1, 0.1])
    parser.add_argument("--grid-max-train_samples", type=int, default=None)
    parser.add_argument("--grid-max-val-samples", type=int, default=None)
    parser.add_argument("--grid-max-test-samples", type=int, default=None)
    
    # ... 나머지 인자들도 그대로 유지 (생략 없이 전체 코드가 필요하면 기존 파일 내용 사용) ...
    # 아래는 편의상 기존 parse_args 내용이 있다고 가정하고 생략합니다.
    # 실제 파일 수정 시에는 기존 parse_args 함수 내용을 그대로 두세요.
    
    # (여기서는 코드가 너무 길어지는 것을 방지하기 위해 위쪽 원본 코드의 parse_args를 그대로 쓴다고 가정)
    # 원본 코드의 parse_args() 복사해서 사용하세요.
    
    # 여기서는 임시로 필요한 부분만 다시 적지 않고, 
    # 사용자가 제공한 파일의 parse_args를 그대로 사용하면 됩니다.
    return parser.parse_args() 

# 만약 parse_args 함수 전체가 필요하다면, 
# 사용자가 제공한 `train.py`의 `parse_args` 함수를 그대로 사용하세요. 
# 위 코드 블록에서는 중복을 피하기 위해 생략했습니다.
# 아래에 제공해주신 train.py의 parse_args를 그대로 넣으세요.
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--exp-dir", default=None, type=str)
    parser.add_argument("--root-dir", default=None, type=str)
    parser.add_argument("--dataset-name", default="lrs3", type=str, choices=["lrs3", "grid"])
    parser.add_argument("--grid-split-ratio", nargs=3, type=float, default=[0.8, 0.1, 0.1])
    parser.add_argument("--grid-max-train-samples", type=int, default=None)
    parser.add_argument("--grid-max-val-samples", type=int, default=None)
    parser.add_argument("--grid-max-test-samples", type=int, default=None)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--project-wandb", default=None, type=str)
    parser.add_argument("--exp-name", default="", type=str)
    parser.add_argument("--modality", default=None, type=str, choices=["audio", "video", "audiovisual", "audiovisual_avhubert"])
    parser.add_argument("--llm-model", default=None, type=str)
    parser.add_argument("--intermediate-size", default=2048, type=int)
    parser.add_argument("--single-projector-avhubert", default=False, type=bool)
    parser.add_argument("--prompt-audio", default="Transcribe speech to text.", type=str)
    parser.add_argument("--prompt-video", default="Transcribe video to text.", type=str)
    parser.add_argument("--prompt-audiovisual", default="Transcribe speech and video to text.", type=str)
    parser.add_argument("--pretrain-avhubert-enc-video-path", default=None, type=str)
    parser.add_argument("--pretrain-avhubert-enc-audio-path", default=None, type=str)
    parser.add_argument("--pretrain-avhubert-enc-audiovisual-path", default=None, type=str)
    parser.add_argument("--use-lora-avhubert", default=False, type=bool)
    parser.add_argument("--audio-encoder-name", default=None, type=str)
    parser.add_argument("--unfrozen_modules", nargs="*", default=[None])
    parser.add_argument("--add_PETF_LLM", default=None, type=str)
    parser.add_argument("--reduction_lora", default=None, type=int)
    parser.add_argument("--alpha", default=None, type=int)
    parser.add_argument("--downsample-ratio-audio", default=3, type=int)
    parser.add_argument("--downsample-ratio-video", default=3, type=int)
    parser.add_argument("--downsample-ratio-audiovisual", default=2, type=int)
    parser.add_argument("--train-file", default="lrs3_train.csv", type=str)
    parser.add_argument("--val-file", default="lrs3_test.csv", type=str)
    parser.add_argument("--test-file", default="lrs3_test.csv", type=str)
    parser.add_argument("--num-nodes", default=1, type=int)
    parser.add_argument("--gpus", default=1, type=int)
    parser.add_argument("--pretrained-model-path", default=None, type=str)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--max-epochs", default=10, type=int)
    parser.add_argument("--num-average-epochs", default=4, type=int)
    parser.add_argument("--num-check-save", default=4, type=int)
    parser.add_argument("--val-check-interval", default=1.0) # float로 수정
    parser.add_argument("--max-frames-audio", type=int, default=1000)
    parser.add_argument("--max-frames-video", type=int, default=1500)
    parser.add_argument("--max-frames-audiovisual", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--train-num-buckets", type=int, default=400)
    parser.add_argument("--ckpt-path", type=str, default=None)
    parser.add_argument("--max-dec-tokens", default=32, type=int)
    parser.add_argument("--num-beams", default=15, type=int)
    parser.add_argument("--slurm-job-id", type=float, default=-1)
    parser.add_argument("--decode-snr-target", type=float, default=999999)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--auto-test", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)
    # UADF 관련 옵션 추가
    parser.add_argument("--use-uadf", action="store_true", help="UADF 사용 여부")
    parser.add_argument("--uadf-fusion-method", type=str, default="uncertainty", choices=["uncertainty", "attention"])
    parser.add_argument("--uadf-temperature", type=float, default=1.0)
    
    return parser.parse_args()

def cli_main():
    args = parse_args()
    print("Selected Dataset:", args.dataset_name)
    print("Root Directory:", args.root_dir)
    
    if args.slurm_job_id != -1:
        args.slurm_job_id = os.environ["SLURM_JOB_ID"]
    
    modelmodule = ModelModule_LLM(args)

    # [수정 3] dataset-name에 따라 Datamodule 선택 분기 추가
    if args.dataset_name == "grid":
        print("Initializing GRID DataModule...")
        datamodule = GRIDDataModule(args, modelmodule.tokenizer)
    else:
        # 기존 로직 유지
        datamodule = DataModule_LLM(args, modelmodule.tokenizer, train_num_buckets=args.train_num_buckets)

    trainer = get_trainer(args)
    trainer.fit(model=modelmodule, datamodule=datamodule, ckpt_path=args.ckpt_path)
    trainer.print(torch.cuda.memory_summary())
    
    if args.auto_test:
        args.pretrained_model_path = ensemble_original(args, args.num_average_epochs)
        time.sleep(600)
        torch.distributed.destroy_process_group()
        if trainer.is_global_zero:
            trainer = get_test_trainer(args)
            ckpt = torch.load(args.pretrained_model_path, map_location=lambda storage, loc: storage)
            modelmodule.model.load_state_dict(ckpt)
            trainer.test(model=modelmodule, datamodule=datamodule)

if __name__ == "__main__":
    cli_main()