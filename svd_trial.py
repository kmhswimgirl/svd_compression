import cv2
import matplotlib.pyplot as plt
import numpy as np

img = 'aviatabors.jpg'
img_data = cv2.imread(img)

# Return the current image
def show_image(image, title:str):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# for color images / output
def svd_color(image):
    B, G, R = cv2.split(image)

    U_b, S_b, Vt_b = np.linalg.svd(B, full_matrices=False)
    U_g, S_g, Vt_g = np.linalg.svd(G, full_matrices=False)
    U_r, S_r, Vt_r = np.linalg.svd(R, full_matrices=False)

    n = 5  # number of singular values
    R_comp = np.matrix(U_r[:, :n]) * np.diag(S_r[:n]) * np.matrix(Vt_r[:n, :])
    G_comp = np.matrix(U_g[:, :n]) * np.diag(S_g[:n]) * np.matrix(Vt_g[:n, :])
    B_comp = np.matrix(U_b[:, :n]) * np.diag(S_b[:n]) * np.matrix(Vt_b[:n, :])
    
    plt.subplot(1, 3, 1)
    plt.title('Red')
    plt.imshow(R_comp, cmap='Reds_r')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(B_comp, cmap='Blues_r')
    plt.title('Blue')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(G_comp, cmap='Greens_r')
    plt.title('Green')
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

svd_color(img_data)