import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import h5py
import scipy.io as sio
import kagglehub

# ---------------------------------------------------------
# 1. RÉCUPÉRATION DU DATASET
# ---------------------------------------------------------
print("Vérification du dataset Kaggle...")
path = kagglehub.dataset_download("simhadrisadaram/icvl-test-512")
mat_files = glob.glob(os.path.join(path, "**", "*.mat"), recursive=True)
mat_files.sort() # Pour avoir un ordre logique

if len(mat_files) == 0:
    print("Aucun fichier .mat n'a été trouvé.")
    exit()

print(f"{len(mat_files)} images trouvées. Lancement de l'interface...")

# ---------------------------------------------------------
# 2. FONCTION DE CHARGEMENT ROBUSTE
# ---------------------------------------------------------
def load_pseudo_rgb(file_path):
    """Charge un .mat à la volée et renvoie une image RGB."""
    try:
        with h5py.File(file_path, "r") as f:
            keys = list(f.keys())
            key = 'rad' if 'rad' in keys else keys[0]
            image_np = np.array(f[key], dtype=np.float32)
    except OSError:
        mat_data = sio.loadmat(file_path)
        keys = [k for k in mat_data.keys() if not k.startswith('_')]
        key = 'rad' if 'rad' in keys else keys[0]
        image_np = mat_data[key].astype(np.float32)

    if image_np.shape[-1] == 31 or image_np.shape[-1] > 100:
        image_np = np.transpose(image_np, (2, 0, 1))

    nb_canaux = image_np.shape[0]
    
    if nb_canaux >= 31:
        R, G, B = image_np[27, :, :], image_np[14, :, :], image_np[8, :, :]
    else:
        R = image_np[int(nb_canaux * 0.9), :, :]
        G = image_np[int(nb_canaux * 0.5), :, :]
        B = image_np[int(nb_canaux * 0.1), :, :]

    rgb = np.stack([R, G, B], axis=-1)
    
    rgb_min, rgb_max = rgb.min(), rgb.max()
    if rgb_max > rgb_min:
        rgb = (rgb - rgb_min) / (rgb_max - rgb_min)
        
    return np.clip(rgb * 1.5, 0, 1)

# ---------------------------------------------------------
# 3. INTERFACE GRAPHIQUE INTERACTIVE
# ---------------------------------------------------------
# Création de la figure avec un peu d'espace en bas pour le slider
fig, ax = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(bottom=0.2)

# Premier affichage (Image 0)
current_idx = 0
rgb_initial = load_pseudo_rgb(mat_files[current_idx])
img_display = ax.imshow(rgb_initial)
ax.set_title(f"[{current_idx + 1}/{len(mat_files)}] - {os.path.basename(mat_files[current_idx])}")
ax.axis('off')

# Création de l'axe et du Slider en bas de la fenêtre
ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
slider = Slider(
    ax=ax_slider,
    label='Image n°',
    valmin=0,
    valmax=len(mat_files) - 1,
    valinit=current_idx,
    valstep=1
)

# Fonction appelée à chaque fois qu'on bouge le curseur
def update(val):
    idx = int(slider.val)
    try:
        # On charge la nouvelle image
        new_rgb = load_pseudo_rgb(mat_files[idx])
        # On met à jour l'affichage sans recréer toute la fenêtre
        img_display.set_data(new_rgb)
        ax.set_title(f"[{idx + 1}/{len(mat_files)}] - {os.path.basename(mat_files[idx])}")
        fig.canvas.draw_idle()
    except Exception as e:
        print(f"Erreur lors du chargement de l'image {idx}: {e}")

# Lier le slider à la fonction de mise à jour
slider.on_changed(update)

plt.show()