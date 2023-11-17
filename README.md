# A Single Image to Consistent Multi-view Images
The code transform a single image into consistent multi-view images. SAM is employed for precise object segmentation and background removal, while Zero123++ leverages diffusion-based techniques for generating diverse views.
## SAM (Segment Anything)
The Segment Anything Model (SAM) produces high quality object masks from input prompts such as points or boxes, and it can be used to generate masks for all objects in an image. It has been trained on a dataset of 11 million images and 1.1 billion masks, and has strong zero-shot performance on a variety of segmentation tasks.

## Generating Multi-view Images Using Zero123plus
Multi-view Generation: The gen_multiview function uses the Zero123++ diffusion pipeline to generate consistent multi-view images.
Zero123 is a versatile framework designed for altering the camera viewpoint of an object using only a single RGB image. Operating in an under-constrained environment, the framework excels in synthesizing novel views by leveraging geometric priors learned from large-scale diffusion models. Utilizing a conditional diffusion model trained on a synthetic dataset, Zero123 learns the controls governing relative camera viewpoints. This knowledge allows the generation of new images depicting the same object under specified camera transformations. Despite being trained on synthetic data, Zero123 showcases robust zero-shot generalization capabilities, extending its effectiveness to out-of-distribution datasets and diverse real-world images, including unconventional cases like impressionist paintings. Beyond view synthesis, Zero123's viewpoint-conditioned diffusion approach proves valuable for 3D reconstruction from a single image. Rigorous qualitative and quantitative experiments demonstrate the framework's superiority over state-of-the-art models in tasks such as single-view 3D reconstruction and novel view synthesis, thanks to its leveraging of Internet-scale pre-training.

To generate multi-view images from a single input image using Zero123++ diffusion pipeline, you can run the following code

<pre>
import torch
import requests
from PIL import Image
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

# Load the pipeline
pipeline = DiffusionPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.1", custom_pipeline="sudo-ai/zero123plus-pipeline",
    torch_dtype=torch.float16
)
pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipeline.scheduler.config, timestep_spacing='trailing'
)
pipeline.to('cuda:0')
cond = Image.open(requests.get("https://d.skis.ltd/nrp/sample-data/lysol.png", stream=True).raw)

# Run the pipeline!
result = pipeline(cond, num_inference_steps=75).images[0]
result.show()
result.save("output.png")
</pre>

# Usage

## Dependencies:
Before running the code, make sure to install the required dependencies by running:

`pip install -r requirements.txt`

The requirements.txt file specifies the Python packages and their versions required to run the code. Follwoing are the main libraries used:
* rembg: rembg library for background removal in images using the alpha matting technique.
* segment-anything: A Git repository containing the code for the segment-anything library, likely used for segmentation tasks.
* transformers: Hugging Face's Transformers library, which provides pre-trained models and utilities for natural language processing.


Additionally, download the required checkpoint for SAM file by running:
`python download_checkpoints.py`
This downloads the checkpoints in the tmp folder that are used by segment-anything.

Then run 
`main.py <input_image_path> [<output_folder>].`

Parameters/Arguments:

* <input_image_path>: Replace this placeholder with the path to the input image that you want to process. This is the image for which you want to generate consistent multi-view images.

* [<output_folder>]: This part is optional. If specified, it represents the folder where the generated multi-view images will be saved. If not provided, it will create a folder named 'output'.

