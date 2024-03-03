from argparse import Namespace
import sys
import os
import torch
import numpy as np
from pathlib import Path
from accelerate import Accelerator
from torchvision import transforms
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from otter.modeling_otter import OtterForConditionalGeneration
from otter.biovil_encoder.image.data.io import load_image
from otter.biovil_encoder.image.data.transforms import create_chest_xray_transform_for_inference
from med_datasets.input_dataset import pad_or_cut_img_tensors



class GeneratorOtter():
    def __init__(self, cfg):
        self.cfg = cfg
        self.accelerator = Accelerator(mixed_precision=self.cfg.precision)
        self.model = OtterForConditionalGeneration.from_pretrained(
            "luodian/openflamingo-9b-hf",
            device_map="auto",
            vision_encode_mode=self.cfg.vision_encode_mode,
            downsample_frames=self.cfg.downsample_frame,
            num_vision_tokens=self.cfg.num_vision_token
        )
        self.model.text_tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<|endofchunk|>", "<image>", "<answer>"]}
        )
        self.model.lang_encoder.resize_token_embeddings(len(self.model.text_tokenizer))
        self.model.init_medical_vision_encoder(self.cfg)
        self.model.init_medical_roi_extractor(self.cfg)
        cwd = os.getcwd()
        ckpt = torch.load(os.path.join(cwd, "ChatCAD", self.cfg.pretrained_model_name_or_path), map_location=self.model.device)
        if "model_state_dict" in ckpt.keys():
            ckpt = ckpt["model_state_dict"]
        missing_keys, unexpected_keys = self.model.load_state_dict(ckpt, strict=False)
        loaded_keys = list(ckpt.keys())
        for key in unexpected_keys:
            print(key)
            loaded_keys.remove(key)
        print(loaded_keys)
        if self.cfg.vision_encode_mode == "original":
           assert len(unexpected_keys) == 0
           for key in missing_keys:
               assert "vision_encoder" in key or "lang_encoder" in key
        elif "llama_adapter" in self.cfg.vision_encode_mode:
            assert len(unexpected_keys) == 0, unexpected_keys
            for key in missing_keys:
                assert "adapter" not in key

        self.model.text_tokenizer.padding_side = "left"

        self.med_patch_resize_transform = create_chest_xray_transform_for_inference(
            resize=512, center_crop_size=self.cfg.med_patch_image_size
        )
        self.resize_transform = transforms.Resize(self.cfg.patch_image_size, antialias=True)

    def report(self, image_path):

        image = load_image(image_path)
        med_image = self.med_patch_resize_transform(image)
        image = self.resize_transform(med_image)
        image = pad_or_cut_img_tensors(
            torch.stack([image]),
            self.cfg.patch_image_size,
            self.cfg.num_images_per_sample
        ).to(self.model.device).unsqueeze(0).unsqueeze(2)
        med_image = pad_or_cut_img_tensors(
            torch.stack([med_image]),
            self.cfg.med_patch_image_size,
            self.cfg.num_images_per_sample
        ).to(self.model.device).unsqueeze(0).unsqueeze(2)
        text = f"<s> <image>User: Act as a radiologist and write a diagnostic radiology report for the patient based on their chest radiographs: GPT:<answer> this is gt <|endofchunk|>"
        input_ids = self.model.text_tokenizer(
            f"{text}",
            return_tensors="pt",
            add_special_tokens=False,
            max_length=self.cfg.max_length
        )["input_ids"]
        input_ids_np = input_ids.cpu().numpy()
        answer_token_id = self.model.text_tokenizer("<answer>", add_special_tokens=False)["input_ids"][-1]
        answer_indices = list(np.where(input_ids_np == answer_token_id)[-1])
        assert len(answer_indices) == 1, "Current only support batch size = 1"
        input_ids, gt_ids = torch.tensor_split(input_ids, answer_indices, dim=-1)
        input_ids = torch.concat((input_ids, torch.tensor([[answer_token_id]]).to(input_ids.device)), dim=-1)
        gt_ids = gt_ids[:, 1:-2]

        with torch.no_grad() and self.accelerator.autocast():
            # start = time.time()
            generated_text = self.model.generate(
                vision_x=image,
                lang_x=input_ids.to(self.model.device),
                med_vision_x=med_image,
                attention_mask=torch.ones(input_ids.shape, dtype=torch.int32, device=self.model.device),
                max_new_tokens=512,
                num_beams=self.cfg.n_beams,
                label_only=False
            )
            # print(time.time()-start)

        gt = self.model.text_tokenizer.decode(gt_ids[0]).strip("\n\t ").lower()
        pred = self.model.text_tokenizer.decode(generated_text[0]).strip("\n\t ")
        pred = pred[:pred.find("</s>")] if pred.find("</s>") > 0 else pred
        pred = pred[pred.find("<answer>"):] if pred.find("<answer>") > 0 else pred
        pred = pred.replace("<answer>", "").replace("<|endofchunk|>", "").strip("\n\t ").lower()
        return pred