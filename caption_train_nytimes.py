import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
seed = 42
# os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化
# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # 在cuda 10.2及以上的版本中，需要设置以下环境变量来保证cuda的结果可复现

from pprint import pprint
from lightning.pytorch import seed_everything
import torch
import random
import numpy as np
# 模型初始化结束再设置随机种子
seed_everything(seed, workers=True)
def setup_seed(seed):
	#  下面两个常规设置了，用来np和random的话要设置 
    np.random.seed(seed) 
    random.seed(seed)
    
    os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # 在cuda 10.2及以上的版本中，需要设置以下环境变量来保证cuda的结果可复现
    
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # 多GPU训练需要设置这个
    torch.manual_seed(seed)
    
    torch.use_deterministic_algorithms(True) # 一些操作使用了原子操作，不是确定性算法，不能保证可复现，设置这个禁用原子操作，保证使用确定性算法
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.enabled = False  # 禁用cudnn使用非确定性算法
    torch.backends.cudnn.benchmark = False  # 与上面一条代码配套使用，True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现。
setup_seed(seed)

from configs.config_nytimes import parser
from dataset.goodnews_module import DataModule
from lightning_tools.callbacks import add_callbacks
from models.CaptionGenGPT import CaptionGenGPT
import lightning.pytorch as pl

def train(args):
    dm = DataModule(args)
    callbacks = add_callbacks(args)

    trainer = pl.Trainer(
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=args.strategy,
        accelerator=args.accelerator,
        precision=args.precision,
        val_check_interval = args.val_check_interval,
        limit_val_batches = args.limit_val_batches,
        max_epochs = args.max_epochs,
        num_sanity_val_steps = args.num_sanity_val_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        sync_batchnorm=True,
        callbacks=callbacks["callbacks"], 
        logger=callbacks["loggers"]
    )
    if args.ckpt_file is not None:
        model = CaptionGenGPT.load_from_checkpoint(args.ckpt_file, strict=False, args=args, map_location="cpu")
    else:
        model = CaptionGenGPT(args)

    if args.test:
        trainer.test(model, datamodule=dm)
    elif args.validate:
        trainer.validate(model, datamodule=dm)
    else:
        trainer.fit(model, datamodule=dm)

def main():
    args = parser.parse_args()
    
    os.makedirs(args.savedmodel_path, exist_ok=True)
    pprint(vars(args))
    seed_everything(42, workers=True)
    # args.validate = True
    train(args)


if __name__ == '__main__':
    main()