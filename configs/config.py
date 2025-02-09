import argparse

parser = argparse.ArgumentParser(description="hyper-parameter for InstructBlip")
# ========================= Dataset Configs ==========================
parser.add_argument('--host', default="localhost", type=str, help="mongodb host")
parser.add_argument('--port', default="27017", type=str, help="mongodb port")
parser.add_argument('--test', action='store_true', help="only run test set")
parser.add_argument('--validate', action='store_true', help="only run validation set")
parser.add_argument('--dataset', type=str, default='mimic_cxr', help="goodnews or NYTimes800k")
parser.add_argument('--annotation', type=str, default=r'goodnews.json', help="annotation file of the dataset")
parser.add_argument('--base_dir', type=str, default=r'dataset/goodnews/images', help="base dir to help find images")
parser.add_argument('--batch_size', default=1, type=int, help="use for training duration per worker")
parser.add_argument('--val_batch_size', default=1, type=int, help="use for validation duration per worker")
parser.add_argument('--test_batch_size', default=2, type=int, help="use for testing duration per worker")
parser.add_argument('--prefetch_factor', default=4, type=int, help="use for training duration per worker")
parser.add_argument('--num_workers', default=8, type=int, help="Cpu num for dataloaders")

# ========================= Model Settings ============================
parser.add_argument('--vllm', default='instructblip-vicuna-7b', type=str, help="VLLM model to use")
parser.add_argument('--end_sym', default='</s>', type=str)

# ======================== SavedModel Configs ===========================
parser.add_argument('--savedmodel_path', type=str, default='save/goodnews/train_total')
parser.add_argument('--ckpt_file', type=str, default=None, help='the checkpoint file to load')
parser.add_argument('--delta_file', type=str, default=None, help='the delta file to load')
parser.add_argument('--weights', type=list, default=[0.5, 0.5])
parser.add_argument('--scorer_types', type=list, default=['Bleu_4', 'CIDEr'])

# ========================= Learning Configs ==========================
parser.add_argument('--learning_rate', default=2e-5, type=float, help='initial learning rate')
parser.add_argument('--gradient_clip_val', default=None, type=int, help='gradient clip value')

# ========================= Decoding Settings ==========================
parser.add_argument('--beam_size', type=int, default=5)
parser.add_argument('--do_sample', type=bool, default=False)
parser.add_argument('--no_repeat_ngram_size', type=int, default=2)
parser.add_argument('--num_beam_groups', type=int, default=1)
parser.add_argument('--min_new_tokens', type=int, default=5)
parser.add_argument('--max_new_tokens', type=int, default=50)
parser.add_argument('--max_length', type=int, default=50)
parser.add_argument('--repetition_penalty', type=float, default=2.0)
parser.add_argument('--length_penalty', type=float, default=2.0)
parser.add_argument('--diversity_penalty', type=float, default=0)
parser.add_argument('--temperature', type=float, default=0.0)

# ====================== Pytorch Lightning ===========================
parser.add_argument('--devices', type=int, default=4, help='how many gpus to use')
parser.add_argument('--num_nodes', type=int, default=1, help='Number of GPU nodes for distributed training.')
parser.add_argument('--accelerator', type=str, default="gpu", choices=["cpu", "gpu", "tpu", "ipu", "hpu", "mps"], help='accelerator types')
parser.add_argument('--strategy', type=str, default="deepspeed_stage_2", help='default ddp for multi-gpus')
parser.add_argument('--precision', type=str, default='bf16-mixed', help='16 or 32 bf16-mixed, using for original pytorch amp auto cast')
parser.add_argument('--limit_val_batches', type=float, default=1.0, help='How much of validation dataset to check (float = fraction, int = num_batches).')
parser.add_argument('--limit_test_batches', type=float, default=1.0, help='How much of test dataset to check (float = fraction, int = num_batches).')
parser.add_argument('--limit_train_batches', type=float, default=1.0, help='How much of training dataset to check (float = fraction, int = num_batches)')
parser.add_argument('--max_epochs', type=int, default=5, help='Stop training once this number of epochs is reached')
parser.add_argument('--every_n_train_steps', type=int, default=0, help='How many training steps to save a checkpoint')
parser.add_argument('--val_check_interval', type=float, default=0.5, help='How often to check the validation set')
parser.add_argument('--accumulate_grad_batches', type=int, default=4, help='Accumulates gradients over k batches before stepping the optimizer')
parser.add_argument("--num_sanity_val_steps", type=int, default=2, help='Sanity check runs n validation batches before starting the training routine')
