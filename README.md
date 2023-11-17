# A Single Image to Consistent Multi-view Images

The code transform a single image into consistent multi-view images. SAM is employed for precise object segmentation and background removal, while Zero123++ leverages diffusion-based techniques for generating diverse views.
## SAM (Segment Anything)
Initialization: The sam_init function loads the SAM model with a specified checkpoint.
Segmentation: sam_segment utilizes SAM to segment an input image based on provided bounding box coordinates, removing the background.
## Image Preprocessing
Rescaling and Segmentation: The preprocess function rescales and segments the input image, preparing it for further processing.
Background Removal: SAM is applied for background removal when specified.
## Generating Multi-view Images Using Zero123plus
Multi-view Generation: The gen_multiview function uses the Zero123++ diffusion pipeline to generate consistent multi-view images.

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

## Usage
Before running the code, make sure to install the required dependencies by running:

`pip install -r requirements.txt`

Additionally, download the required checkpoint file by running:

`python download_checkpoints.py`

Then run `main.py <input_image_path> [<output_folder>].`
