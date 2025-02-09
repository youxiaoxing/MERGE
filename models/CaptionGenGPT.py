import os
import sys
sys.path.append("Merge")
import json
import torch
import torch.nn as nn
import lightning.pytorch as pl
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration, AutoTokenizer
import torch.distributed as dist
import types
from pycocoevalcap.bleu.bleu_scorer import BleuScorer
from pycocoevalcap.cider.cider_scorer import CiderScorer
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
import numpy as np
import re
from tqdm import tqdm
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, TaskType
import gc

from accelerate.hooks import AlignDevicesHook

def remove_hook_from_module(module: torch.nn.Module, recurse=False, hook_cls=AlignDevicesHook):

    if hasattr(module, "_hf_hook") and isinstance(module._hf_hook, hook_cls):
        module._hf_hook.detach_hook(module)
        delattr(module, "_hf_hook")

        if hasattr(module, "_old_forward"):
            module.forward = module._old_forward
            delattr(module, "_old_forward")

    if recurse:
        for child in module.children():
            remove_hook_from_module(child, recurse)

    return module

class CaptionGenGPT(pl.LightningModule):
    """
    R2GenGPT model.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)
        self.processor = InstructBlipProcessor.from_pretrained(args.vllm)
        self.tokenizer = AutoTokenizer.from_pretrained(args.vllm)
        self.qforemr_tokneizer = AutoTokenizer.from_pretrained(args.vllm + "/qformer_tokenizer")
        # 设置8-bit量化配置
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
        # self.model = InstructBlipForConditionalGeneration.from_pretrained(args.vllm, torch_dtype=torch.bfloat16, trust_remote_code=True, ignore_mismatched_sizes=True)
        self.model = InstructBlipForConditionalGeneration.from_pretrained(args.vllm, torch_dtype=torch.bfloat16, trust_remote_code=True)

        self.model.language_model = InstructBlipForConditionalGeneration.from_pretrained(
            args.vllm,
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            # device_map="cpu"
        ).language_model.cpu()
        
        remove_hook_from_module(self.model, recurse=True)
        
        peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, inference_mode=False, r=16, lora_alpha=16, lora_dropout=0.05,
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            )
        
        # self.model.language_model._set_gradient_checkpointing(True)
        # self.model.vision_model._set_gradient_checkpointing(True)
        # self.model.qformer._set_gradient_checkpointing(True)
        # self.model.gat._set_gradient_checkpointing(True)
        
        for name, param in self.model.vision_model.named_parameters():
            param.requires_grad = False
        
        for name, param in self.model.language_model.named_parameters():
            param.requires_grad = False

        self.model.language_model = prepare_model_for_kbit_training(
            self.model.language_model,
            use_gradient_checkpointing=True
        )
        self.model.language_model = get_peft_model(self.model.language_model, peft_config)
        
        # self.model.print_trainable_parameters()
        self.qformer_txt_len = 512
        self.max_txt_len = 1024
        self.max_txt_len_valuation = 800
        self.max_output_txt_len = 50
        self.val_step_outputs = []
        self.val_step_ref = []
        self.val_step_ids = []
        torch.cuda.empty_cache()
        gc.collect()
    
    def cal_caption_score_from_dict(self, result_dict):
        bleu_scorer = BleuScorer(n=4)
        rouge_scorer = Rouge()
        rouge_scores = []
        cider_scorer = CiderScorer(n=4, sigma=6.0)
        meteor_scorer = Meteor()
        meteor_scorer._stat = types.MethodType(_stat, meteor_scorer)

        eval_line = 'EVAL'
        meteor_scorer.lock.acquire()
        count = 0
        meteor_scores = []

        for sample in tqdm(result_dict):
            # Remove punctuation
            caption = re.sub(r'[^\w\s]', '', sample["ref_caption"])
            generation = re.sub(r'[^\w\s]', '', sample["caption"])

            bleu_scorer += (generation, [caption])
            rouge_score = rouge_scorer.calc_score([generation], [caption])
            rouge_scores.append(rouge_score)
            cider_scorer += (generation, [caption])

            stat = meteor_scorer._stat(generation, [caption])
            eval_line += ' ||| {}'.format(stat)
            count += 1

        meteor_scorer.meteor_p.stdin.write('{}\n'.format(eval_line).encode())
        meteor_scorer.meteor_p.stdin.flush()
        for _ in range(count):
            meteor_scores.append(float(meteor_scorer.meteor_p.stdout.readline().strip()))
        meteor_score = float(meteor_scorer.meteor_p.stdout.readline().strip())
        meteor_scorer.lock.release()

        blue_score, _ = bleu_scorer.compute_score(option='closest')
        rouge_score = np.mean(np.array(rouge_scores))
        cider_score, _ = cider_scorer.compute_score()

        return {
            'Bleu_1': float(blue_score[0]),
            'Bleu_2': float(blue_score[1]),
            'Bleu_3': float(blue_score[2]),
            'Bleu_4': float(blue_score[3]),
            'ROUGE': float(rouge_score),
            'CIDEr': float(cider_score),
            'METEOR': float(meteor_score)
        }


    def concat_text_input_output(self, input_ids, input_atts, output_ids, output_atts):
        input_part_targets_len = []
        llm_tokens = {"input_ids": [], "attention_mask": []}
        for i in range(input_ids.size(0)):
            this_input_ones = input_atts[i].sum()
            input_part_targets_len.append(this_input_ones)
            llm_tokens['input_ids'].append(
                torch.cat([
                    input_ids[i][:this_input_ones],
                    output_ids[i][1:],
                    input_ids[i][this_input_ones:]
                ])
            )
            llm_tokens['attention_mask'].append(
                torch.cat([
                    input_atts[i][:this_input_ones],
                    output_atts[i][1:],
                    input_atts[i][this_input_ones:]
                ])
            )
        llm_tokens['input_ids'] = torch.stack(llm_tokens['input_ids'])
        llm_tokens['attention_mask'] = torch.stack(llm_tokens['attention_mask'])
        return llm_tokens, input_part_targets_len

    def forward(self, samples):
        image = samples["images"]
        captions = samples["captions"]
        captions = [text + self.tokenizer.eos_token for text in captions]
        contexts = samples["contexts"]
        _ids = samples["_ids"]
        graph_str = samples["graph_str"]
        ### pixel_values
        pixel_values = image
        
        ### qformer_input_ids和mask attention
        qformer_text_tokens = self.qforemr_tokneizer(
            contexts,
            padding='longest',
            truncation=True,
            max_length=self.qformer_txt_len,
            return_tensors="pt",
        ).to(image.device)
        
        qformer_input_ids = qformer_text_tokens.input_ids
        qformer_attention_mask = qformer_text_tokens.attention_mask
        
        ### input_ids 和 attention_mask
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = 'left'
        context_text_tokens = self.tokenizer(
            contexts,
            padding='longest',
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)
        

        self.tokenizer.truncation_side = 'right'
        captions_tokens = self.tokenizer(
            captions,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_output_txt_len,
        ).to(image.device)
        
        llm_tokens, input_part_targets_len = self.concat_text_input_output(
            context_text_tokens.input_ids,
            context_text_tokens.attention_mask,
            captions_tokens.input_ids,
            captions_tokens.attention_mask,
        )
        
        # do not apply loss to the padding
        targets = llm_tokens['input_ids'].masked_fill(
            llm_tokens['input_ids'] == self.tokenizer.pad_token_id, -100
        )

        # do not apply loss to the text input (i.e., instruction)
        for i, l in enumerate(input_part_targets_len):
            targets[i][:l] = -100
            
        outptus = self.model(
            pixel_values=pixel_values,
            qformer_input_ids = qformer_input_ids,
            qformer_attention_mask = qformer_attention_mask,
            input_ids=llm_tokens["input_ids"],
            attention_mask = llm_tokens["attention_mask"],
            labels = targets,
            return_dict=True,
            graph_str = graph_str
        )
        
        loss = outptus.loss
        return {"loss": loss}
        

    def training_step(self, batch, batch_idx):
        result = self(batch)
        self.log_dict(result, prog_bar=True)
        return result

    def save_checkpoint(self, eval_res=None):
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        param_grad_dic = {
            k: v.requires_grad for (k, v) in self.named_parameters() if v.requires_grad
        }
        state_dict = self.state_dict()
        # for k in list(state_dict.keys()):
        #     if k not in param_grad_dic.keys():
        #         del state_dict[k]
        save_obj = {
            "state_dict": state_dict,
            "config": self.hparams,
            "epoch": current_epoch,
            "step":global_step,
            "pytorch-lightning_version": "2.2.1"
        }
        if eval_res is not None:
            os.makedirs(os.path.join(self.hparams.savedmodel_path, 'checkpoints'), exist_ok=True)
            save_to = os.path.join(
                self.hparams.savedmodel_path, 'checkpoints',
                "checkpoint_epoch{}_step{}_bleu{:3f}_cider{:3f}.pth".format(current_epoch, global_step, eval_res['Bleu_4'], eval_res['CIDEr']),
            )
        else:
            os.makedirs(os.path.join(self.hparams.savedmodel_path, 'checkpoints'), exist_ok=True)
            save_to = os.path.join(
                self.hparams.savedmodel_path, 'checkpoints',
                "checkpoint_epoch{}_step{}.pth".format(current_epoch, global_step),
            )
        self.print("Saving checkpoint at step {} to {}.".format(global_step, save_to))
        torch.save(save_obj, save_to)
    
    def validation_step(self, samples, batch_idx):
        image = samples["images"]
        captions = samples["captions"]
        captions = [text + self.tokenizer.eos_token for text in captions]
        contexts = samples["contexts"]
        _ids = samples["_ids"]
        graph_str = samples["graph_str"]
        ### pixel_values
        pixel_values = image
        
        ### qformer_input_ids和mask attention
        qformer_text_tokens = self.qforemr_tokneizer(
            contexts,
            padding='longest',
            truncation=True,
            max_length=self.qformer_txt_len,
            return_tensors="pt",
        ).to(image[0].device)
        
        qformer_input_ids = qformer_text_tokens.input_ids
        qformer_attention_mask = qformer_text_tokens.attention_mask
        
        ### input_ids 和 attention_mask
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = 'left'
        context_text_tokens = self.tokenizer(
            contexts,
            padding='longest',
            truncation=True,
            max_length=self.max_txt_len_valuation,
            return_tensors="pt",
        ).to(image[0].device)
        
        
        self.tokenizer.truncation_side = 'right'
        to_regress_tokens = self.tokenizer(
            captions,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
        ).to(image[0].device)
             
            
        outputs = self.model.generate(
            pixel_values=pixel_values,
            qformer_input_ids = qformer_input_ids,
            qformer_attention_mask = qformer_attention_mask,
            input_ids=context_text_tokens.input_ids,
            attention_mask = context_text_tokens.attention_mask,
            num_beams=self.hparams.beam_size,
            do_sample=self.hparams.do_sample,
            min_new_tokens=self.hparams.min_new_tokens,
            max_new_tokens=self.hparams.max_new_tokens,
            repetition_penalty=self.hparams.repetition_penalty,
            length_penalty=self.hparams.length_penalty,
            temperature=self.hparams.temperature,
            graph_str=graph_str
        )

        # hypo = [self.decode(i) for i in outputs]
        # ref = [self.decode(i) for i in to_regress_tokens["input_ids"]]
        # print(hypo)
        # print(ref)
        # time.sleep(10000)
        # ref = [self.decode(i) for i in to_regress_tokens['input_ids']] ##############
        # self.val_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["ids"]})
        # return hypo, ref
        # self.decode(outputs)
        self.val_step_outputs.append(outputs)
        self.val_step_ref.append(to_regress_tokens["input_ids"])
        self.val_step_ids.append(self.pad_and_convert_to_tensor(_ids))
        # self.test_step_outputs.append(outputs)
        torch.cuda.empty_cache()
        return [outputs, to_regress_tokens["input_ids"], self.pad_and_convert_to_tensor(_ids)]
    
    def truncate_texts(self, input_texts, max_tokens=100):
        truncated_texts = []
        
        for text in input_texts:
            # Split each text by spaces
            tokens = text.split()
            # Truncate to the maximum number of tokens
            truncated_tokens = tokens[:max_tokens]
            # Join the tokens back into a single string with spaces
            truncated_text = ' '.join(truncated_tokens)
            
            truncated_texts.append(truncated_text)
        
        return truncated_texts
    
    def pad_and_convert_to_tensor(self, id_list, max_length=50):
        # Convert each string ID to bytes and pad to max_length
        byte_ids = [s.encode('utf-8') for s in id_list]
        padded_ids = [b.ljust(max_length, b'\0') for b in byte_ids]
        return torch.tensor([list(b) for b in padded_ids], dtype=torch.uint8)
    
    def convert_bytes_to_string(self, byte_list):
        # 将整数列表转换为字节字符串
        byte_string = bytes(byte_list)
        # 去除填充的空字符
        stripped_string = byte_string.rstrip(b'\0')
        # 解码UTF-8
        decoded_string = stripped_string.decode('utf-8')
        return decoded_string
    
    def decode(self, output_token):
        output_token[output_token == 0] = 2 # convert output id 0 to 2 (eos_token_id)
        output_text = self.tokenizer.decode(output_token, skip_special_tokens=True)
        print(output_text)
        output_text = output_text.strip()
        return output_text

    def setup_distributed(self):
        if not dist.is_initialized():
            dist.init_process_group(backend='gloo')  # Initialize with gloo for CPU

    def merge(self,outputs):
        if dist.is_initialized():
            all_rank_outputs = [None for _ in range(dist.get_world_size())]    
            dist.all_gather_object(all_rank_outputs,outputs)
            outputs = [x for y in all_rank_outputs for x in y] ## all_rank_output[i]: i-th batch output 
        single_batch_output_cnt = len(outputs[0])
        ret = [[] for _ in range(single_batch_output_cnt)]
        for idx in range(single_batch_output_cnt):
            for batch in outputs:
                ret[idx].append(batch[idx])
        return ret
    
    def on_validation_epoch_start(self):
        self.save_checkpoint()
    
    def on_validation_epoch_end(self):
        # 在每个进程中保存局部结果
        local_rank = dist.get_rank()
        output_file = f'output_rank_{local_rank}.pt'
        torch.save({
            'outputs': self.val_step_outputs,
            'refs': self.val_step_ref,
            'ids': self.val_step_ids
        }, output_file)
        self.val_step_outputs.clear()
        self.val_step_ids.clear()
        self.val_step_ref.clear()
        torch.cuda.empty_cache()
        self.trainer.strategy.barrier()  # 保证所有进程都保存完毕

        if self.trainer.is_global_zero:
            all_outputs = []
            all_refs = []
            all_ids = []
            for rank in range(dist.get_world_size()):
                output_file = f'output_rank_{rank}.pt'
                data = torch.load(output_file)
                all_outputs.extend(data['outputs'])
                all_refs.extend(data['refs'])
                all_ids.extend(data['ids'])
                os.remove(output_file)  # 清理临时文件
            ref = self.truncate_texts([self.decode(i) for res in all_refs for i in res])
            hypo = [self.decode(i) for res in all_outputs for i in res]
            ids = [self.convert_bytes_to_string(i) for res in all_ids for i in res]
            results = []
            for image_id, caption, ref_caption in zip(ids, hypo, ref):
                results.append({
                    'image_id': image_id,
                    'caption': caption,
                    'ref_caption': ref_caption
                })
                
            eval_res = self.cal_caption_score_from_dict(results)
            
            self.log_dict(eval_res, sync_dist=True, logger=True, rank_zero_only=True)

            result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
            os.makedirs(result_folder, exist_ok=True)
            current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
            json.dump(results, open(os.path.join(result_folder, f"result_{current_epoch}_{global_step}" + '.json'), 'w'))
            # json.dump(refs, open(os.path.join(result_folder, f'refs_{current_epoch}_{global_step}.json'), 'w'))
            self.print(eval_res)

            self.val_step_outputs.clear()
            self.val_step_ids.clear()
            self.val_step_ref.clear()
            self.save_checkpoint(eval_res)

    # TODO 修改test代码正确
    def test_step(self, samples, batch_idx):
        pass


    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, eps=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.hparams.max_epochs, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad()
        
def _stat(self, hypothesis_str, reference_list):
    # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
    hypothesis_str = hypothesis_str.replace('|||', '').replace('  ', ' ')
    score_line = ' ||| '.join(
        ('SCORE', ' ||| '.join(reference_list), hypothesis_str))
    score_line = score_line.replace('\n', '').replace('\r', '')
    self.meteor_p.stdin.write('{}\n'.format(score_line).encode())
    self.meteor_p.stdin.flush()
    return self.meteor_p.stdout.readline().decode().strip()