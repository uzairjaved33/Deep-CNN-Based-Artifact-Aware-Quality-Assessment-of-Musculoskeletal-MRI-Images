import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os

# Point to one .gz file
path = r'D:\KMAR-50K\KMAR-50K\ArtifactData_part1'
file = os.path.join(path, os.listdir(path)[0])  # grabs first file

img = nib.load(file).get_fdata()
print("Shape:", img.shape)

# Show middle slice
mid = img.shape[2] // 2
plt.imshow(img[:, :, mid], cmap='gray')
plt.title(f'Slice {mid} of {img.shape[2]}')
plt.axis('off')
plt.show()