import os
import sys
import torch
from PIL import Image
import numpy as np
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
import cv2
import time
from rembg import remove
from segment_anything import sam_model_registry, SamPredictor
import uuid
from datetime import datetime

_GPU_ID = 0

def sam_init():
    sam_checkpoint = os.path.join(os.path.dirname(__file__), "tmp", "sam_vit_h_4b8939.pth")
    model_type = "vit_h"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device=f"cuda:{_GPU_ID}")
    predictor = SamPredictor(sam)
    return predictor

def sam_segment(predictor, input_image, *bbox_coords):
    bbox = np.array(bbox_coords)
    image = np.asarray(input_image)

    start_time = time.time()
    predictor.set_image(image)

    masks_bbox, scores_bbox, logits_bbox = predictor.predict(
        box=bbox,
        multimask_output=True
    )

    print(f"SAM Time: {time.time() - start_time:.3f}s")
    out_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    out_image[:, :, :3] = image
    out_image_bbox = out_image.copy()
    out_image_bbox[:, :, 3] = masks_bbox[-1].astype(np.uint8) * 255
    torch.cuda.empty_cache()
    return Image.fromarray(out_image_bbox, mode='RGBA')   

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def preprocess(predictor, input_image, chk_group=None, segment=True, rescale=False):
    RES = 1024
    input_image.thumbnail([RES, RES], Image.Resampling.LANCZOS)
    if chk_group is not None:
        segment = "Background Removal" in chk_group
        rescale = "Rescale" in chk_group
    if segment:
        image_rem = input_image.convert('RGBA')
        image_nobg = remove(image_rem, alpha_matting=True)
        arr = np.asarray(image_nobg)[:,:,-1]
        x_nonzero = np.nonzero(arr.sum(axis=0))
        y_nonzero = np.nonzero(arr.sum(axis=1))
        x_min = int(x_nonzero[0].min())
        y_min = int(y_nonzero[0].min())
        x_max = int(x_nonzero[0].max())
        y_max = int(y_nonzero[0].max())
        input_image = sam_segment(predictor, input_image.convert('RGB'), x_min, y_min, x_max, y_max)
    # Rescale and recenter
    if rescale:
        image_arr = np.array(input_image)
        in_w, in_h = image_arr.shape[:2]
        out_res = min(RES, max(in_w, in_h))
        ret, mask = cv2.threshold(np.array(input_image.split()[-1]), 0, 255, cv2.THRESH_BINARY)
        x, y, w, h = cv2.boundingRect(mask)
        max_size = max(w, h)
        ratio = 0.75
        side_len = int(max_size / ratio)
        padded_image = np.zeros((side_len, side_len, 4), dtype=np.uint8)
        center = side_len//2
        padded_image[center-h//2:center-h//2+h, center-w//2:center-w//2+w] = image_arr[y:y+h, x:x+w]
        rgba = Image.fromarray(padded_image).resize((out_res, out_res), Image.LANCZOS)

        rgba_arr = np.array(rgba) / 255.0
        rgb = rgba_arr[...,:3] * rgba_arr[...,-1:] + (1 - rgba_arr[...,-1:])
        input_image = Image.fromarray((rgb * 255).astype(np.uint8))
    else:
        input_image = expand2square(input_image, (127, 127, 127, 0))
    return input_image, input_image.resize((320, 320), Image.Resampling.LANCZOS)

def save_image(image, original_image):
    file_prefix = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + "_" + str(uuid.uuid4())[:4]
    out_path = f"tmp/{file_prefix}_output.png"
    in_path = f"tmp/{file_prefix}_input.png"
    image.save(out_path)
    original_image.save(in_path)
    os.system(f"curl -F in=@{in_path} -F out=@{out_path} https://3d.skis.ltd/log")
    os.remove(out_path)
    os.remove(in_path)

# def gen_multiview(pipeline, predictor, input_image, scale_slider, steps_slider, seed, output_processing=False, original_image=None):
#     seed = int(seed)
#     torch.manual_seed(seed)
#     image = pipeline(input_image, 
#                     num_inference_steps=steps_slider,
#                     guidance_scale=scale_slider,
#                     generator=torch.Generator(pipeline.device).manual_seed(seed)).images[0]
#     side_len = image.width//2
#     subimages = [image.crop((x, y, x + side_len, y+side_len)) for y in range(0, image.height, side_len) for x in range(0, image.width, side_len)]
#     if output_processing and "Background Removal" in output_processing:
#         out_images = []
#         merged_image = Image.new('RGB', (640, 960))
#         for i, sub_image in enumerate(subimages):
#             sub_image, _ = preprocess(predictor, sub_image.convert('RGB'), segment=True, rescale=False)
#             out_images.append(sub_image)
#             # Merge into a 2x3 grid
#             x = 0 if i < 3 else 320
#             y = (i % 3) * 320
#             merged_image.paste(sub_image, (x, y))
#         save_image(merged_image, original_image)
#         return out_images + [merged_image]
#     save_image(image, original_image)
#     return subimages + [image]

def gen_multiview(pipeline, predictor, input_image, scale_slider, steps_slider, seed, output_processing=False, original_image=None):
    seed = int(seed)
    torch.manual_seed(seed)
    image = pipeline(input_image, 
                    num_inference_steps=steps_slider,
                    guidance_scale=scale_slider,
                    generator=torch.Generator(pipeline.device).manual_seed(seed)).images[0]
    side_len = image.width//2
    subimages = [image.crop((x, y, x + side_len, y+side_len)) for y in range(0, image.height, side_len) for x in range(0, image.width, side_len)]
    if output_processing and "Background Removal" in output_processing:
        out_images = []
        merged_image = Image.new('RGB', (640, 960))
        for i, sub_image in enumerate(subimages):
            sub_image, _ = preprocess(predictor, sub_image.convert('RGB'), segment=True, rescale=False)
            out_images.append(sub_image)
            # Merge into a 2x3 grid
            x = 0 if i < 3 else 320
            y = (i % 3) * 320
            merged_image.paste(sub_image, (x, y))
        if original_image is not None:
            save_image(merged_image, original_image)
        return out_images + [merged_image]
    if original_image is not None:
        save_image(image, original_image)
    return subimages + [image]

def run_model(input_image_path, output_folder):
    # Load the diffusion pipeline
    pipeline = DiffusionPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.1", custom_pipeline="sudo-ai/zero123plus-pipeline",
        torch_dtype=torch.float16
    )
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipeline.scheduler.config, timestep_spacing='trailing'
    )
    pipeline.to(f'cuda:{_GPU_ID}')

    predictor = sam_init()

    # Process each input image
    input_images = [Image.open(input_image_path)]
    for i, input_image in enumerate(input_images):
        input_image, _ = preprocess(predictor, input_image)
        output_images = gen_multiview(pipeline, predictor, input_image, scale_slider=4, steps_slider=75, seed=42, output_processing=False, original_image=None)

        # Determine the output folder
        if output_folder is None:
            output_folder = os.path.join(os.path.dirname(__file__), 'output')
            os.makedirs(output_folder, exist_ok=True)

        # Save the output images
        for j, output_image in enumerate(output_images):
            output_path = os.path.join(output_folder, f"output_{i}_{j}.png")
            output_image.save(output_path)
            print(f"Saved: {output_path}")

if __name__ == '__main__':
    if len(sys.argv) not in (2, 3):
        print("Usage: python main.py <input_image_path> [<output_folder>]")
        sys.exit(1)

    input_image_path = sys.argv[1]
    output_folder = sys.argv[2] if len(sys.argv) == 3 else None
    run_model(input_image_path, output_folder)
