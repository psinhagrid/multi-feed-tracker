import requests
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

model_id = "IDEA-Research/grounding-dino-tiny"

# Check for available device: CUDA > MPS > CPU
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print("Using device:", device)

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

image_path = "/Users/psinha/Desktop/test_images/test_image_2.jpg"
image = Image.open(image_path)

text_labels = ["Green SweaterLady"]

inputs = processor(images=image, text=text_labels, return_tensors="pt").to(device)

# Start timing inference
start_time = time.time()

with torch.no_grad():
    outputs = model(**inputs)

results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    threshold=0.4,
    text_threshold=0.3,
    target_sizes=[image.size[::-1]]
)

# End timing inference
end_time = time.time()
inference_time = end_time - start_time
print(f"\nInference time: {inference_time:.3f} seconds ({inference_time*1000:.1f} ms)")

result = results[0]

for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
    box = [round(x, 2) for x in box.tolist()]
    print(f"Detected {label} with confidence {round(score.item(), 3)} at location {box}")

# Draw bounding boxes on the image
fig, ax = plt.subplots(1)
ax.imshow(image)

for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
    if score < 0.4:
        continue

    xmin, ymin, xmax, ymax = box.tolist()

    rect = patches.Rectangle(
        (xmin, ymin),
        xmax - xmin,
        ymax - ymin,
        linewidth=2,
        edgecolor='red',
        facecolor='none'
    )

    ax.add_patch(rect)
    ax.text(
        xmin,
        ymin - 5,
        f"{label}: {score:.2f}",
        color='red',
        fontsize=10,
        backgroundcolor='white'
    )

plt.axis("off")
plt.show()

