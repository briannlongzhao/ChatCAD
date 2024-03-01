import os
from argparse import Namespace
from r2g.models import BaseCMNModel
from r2g.modules.tokenizers import Tokenizer
from r2g.modules.generator import Generator
from r2g.modules.generator_otter import GeneratorOtter


def reportGen():
    cwd=os.apth.dirname(os.path.dirname(os.path.realpath(__file__)))
    cfg={
        "visual_extractor":"resnet101",
        "ann_path":os.path.join(cwd, "r2g/annotation.json"),
        "threshold":10,
        "cmm_dim":512,
        "cmm_size":2048,
        'logit_layers':1,
        "d_model":512,
        "d_ff":512,
        "d_vf":2048,
        "num_layers":3,
        "num_heads":8,
        "drop_prob_lm":0.5,
        "dropout":0.1,
        "max_seq_length":100,
        "bos_idx":0,
        "eos_idx":0,
        "pad_idx":0,
        "use_bn":0,
        "n_gpu":1,
        "topk":32,
        "sample_method":"beam_search",
        "sample_n":1,
        "beam_size":3,
        "temperature":1.0,
        "load":os.path.join(cwd, 'weights/r2gcmn_mimic-cxr.pth'),
        "group_size":1,
        "output_logsoftmax":1,
        "decoding_constraint":0, 
        "block_trigrams":1, 
        }
    tokenizer = Tokenizer(cfg)
    model = BaseCMNModel(cfg, tokenizer)
    generator = Generator(cfg, model)
    return generator


def reportGenOtter():
    cfg = Namespace(
        vision_encode_mode="llama_adapter_concat",
        num_vision_token=225,
        downsample_frame=0,
        dataset_type="mimic_cxr",
        medical_vision_encoder_path="/scratch/xinyangjiang/pretrained/biovil_image_resnet50_proj_size_128.pt",
        pretrained_model_name_or_path="./weights/mimic_cxr_final_weights.pt",
        precision="fp16",
        med_patch_image_size=480,
        n_beams=4,
        max_length=500,
        patch_image_size=224,
        num_images_per_sample=2,
    )
    generator = GeneratorOtter(cfg)
    return generator


