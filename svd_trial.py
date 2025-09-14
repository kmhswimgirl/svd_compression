import cv2 # pyright: ignore[reportMissingImports]
import matplotlib.pyplot as plt
import numpy as np

img = 'aviatabors.jpg'
img_data = cv2.imread(img)

# Return the current image
def show_image(image, title:str, cmap):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# for color images / output
def svd_color(image, sv):
    B, G, R = cv2.split(image)

    U_b, S_b, Vt_b = np.linalg.svd(B, full_matrices=False)
    U_g, S_g, Vt_g = np.linalg.svd(G, full_matrices=False)
    U_r, S_r, Vt_r = np.linalg.svd(R, full_matrices=False)

    R_comp = np.matrix(U_r[:, :sv]) * np.diag(S_r[:sv]) * np.matrix(Vt_r[:sv, :])
    G_comp = np.matrix(U_g[:, :sv]) * np.diag(S_g[:sv]) * np.matrix(Vt_g[:sv, :])
    B_comp = np.matrix(U_b[:, :sv]) * np.diag(S_b[:sv]) * np.matrix(Vt_b[:sv, :])
    
    plt.subplot(1, 4, 1)
    plt.title('Red')
    plt.imshow(R_comp, cmap='Reds_r')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(B_comp, cmap='Blues_r')
    plt.title('Blue')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(G_comp, cmap='Greens_r')
    plt.title('Green')
    plt.axis('off')

    comp_full = cv2.merge([np.clip(R_comp, 1, 255), np.clip(G_comp, 1, 255), np.clip(B_comp, 1, 255)])
    comp_full = comp_full.astype(np.uint8)

    plt.subplot(1, 4, 4)
    plt.imshow(comp_full)
    plt.title('RGB Compressed Image')
    plt.axis('off')
    plt.show()

# for grayscale images / output
def svd_gs(image, sv):
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    U, S, Vt = np.linalg.svd(img, full_matrices=False)
    reconstructed_image = np.matrix(U[:,:sv]) * np.diag(S[:sv]) * np.matrix(Vt[:sv,:])
    plt.imshow(reconstructed_image, cmap='gray')
    plt.title('Reconstructed Image')
    plt.axis('off')
    plt.show()

svd_color(img_data, 5)