import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
