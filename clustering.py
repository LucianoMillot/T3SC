import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

from sklearn.cluster import KMeans
import h5py

from hydra import initialize, compose
from omegaconf import OmegaConf

# Ajouter le dossier courant pour que Python trouve le module 't3sc'
sys.path.append(os.path.abspath('.'))
from t3sc.models.multilayer import MultilayerModel

# ------------------------------------------------------------------
# ÉTAPE 1 : CHARGEMENT DU MODÈLE
# ------------------------------------------------------------------
print("Chargement de la configuration...")
with initialize(config_path="t3sc/config", version_base=None):
    cfg = compose(config_name="config", overrides=[
        "data=icvl",
        "noise=constant",
        "noise.params.sigma=25",
        "model.ckpt=icvl_constant_25.ckpt"
    ])

print("Instanciation du modèle PyTorch...")
model_config = OmegaConf.to_container(cfg.model, resolve=True)
kwargs = model_config.get('params', model_config)

model = MultilayerModel(**kwargs)
model.eval()
print("Modèle chargé et prêt !")

# ------------------------------------------------------------------
# ÉTAPE 2 : L'ESPION (HOOK)
# ------------------------------------------------------------------
alphas_capture = {}

def get_activation(name):
    def hook(model, input, output):
        alphas_capture[name] = output.detach().cpu()
    return hook

model.layers[0].register_forward_hook(get_activation('alpha_1'))


# ------------------------------------------------------------------
# ÉTAPE 3 : CHARGEMENT DE VOTRE IMAGE ICVL (Robuste)
# ------------------------------------------------------------------
print("\nChargement de la vraie image ICVL...")
#chemin_image = "CC_40D_2_1103-0917.mat" # <-- Votre fichier à la racine
#chemin_image = "nachal_0823-1222.mat" # <-- Votre fichier à la racine
chemin_image = "nachal_0823-1214.mat" # <-- Votre fichier à la racine

# Vérification rapide de la taille du fichier
taille_mo = os.path.getsize(chemin_image) / (1024 * 1024)
print(f"Taille du fichier : {taille_mo:.2f} Mo")
if taille_mo < 1.0:
    print("ATTENTION : Le fichier pèse moins de 1 Mo. Il est très probablement corrompu ou c'est une page HTML déguisée !")

try:
    # Tentative 1 : Format MATLAB récent (HDF5)
    with h5py.File(chemin_image, "r") as f:
        image_np = np.array(f["rad"], dtype=np.float32)
    print("Fichier lu avec succès (Format HDF5 via h5py).")

except OSError:
    # Tentative 2 : Format MATLAB classique
    print("Le format n'est pas HDF5, tentative de lecture avec scipy.io...")
    try:
        mat_data = sio.loadmat(chemin_image)
        
        # On liste toutes les variables contenues dans le fichier
        cles = [k for k in mat_data.keys() if not k.startswith('_')]
        print(f"Variables trouvées dans le fichier : {cles}")
        
        if 'rad' in cles:
            nom_variable = 'rad'
        else:
            # S'il n'y a pas 'rad', on prend la plus grande variable trouvée
            nom_variable = cles[0] 
            print(f"La clé 'rad' est absente. Utilisation automatique de '{nom_variable}'")
            
        image_np = mat_data[nom_variable].astype(np.float32)
        print("Fichier lu avec succès (Format classique via scipy.io).")
        
    except Exception as e:
        print(f"Erreur fatale : Impossible de lire le fichier. Détail : {e}")
        sys.exit(1)

# --- CORRECTION DES DIMENSIONS ---
# PyTorch veut (Canaux, Hauteur, Largeur). 
# Selon les datasets, l'image est parfois (H, W, Canaux) ou (Canaux, H, W)
if image_np.shape[-1] == 31 or image_np.shape[-1] > 100: 
    # Le dernier chiffre est le nombre de canaux (ex: 31). On doit faire tourner le cube.
    image_tensor = torch.tensor(image_np, dtype=torch.float32).permute(2, 0, 1)
else:
    # Le cube est déjà dans le bon sens
    image_tensor = torch.tensor(image_np, dtype=torch.float32)

# Normalisation Min-Max (entre 0 et 1)
img_min = image_tensor.min()
img_max = image_tensor.max()
image_tensor = (image_tensor - img_min) / (img_max - img_min)

# --- PRÉPARATION DE LA VISUALISATION (Pseudo-RGB) ---
# Sécurité : on vérifie combien de canaux on a vraiment
nb_canaux = image_tensor.shape[0]
if nb_canaux >= 31:
    R = image_tensor[27, :, :].numpy()
    G = image_tensor[14, :, :].numpy()
    B = image_tensor[8, :, :].numpy()
else:
    # Si c'est un autre dataset avec moins de canaux, on prend le début, milieu et fin
    R = image_tensor[int(nb_canaux*0.9), :, :].numpy()
    G = image_tensor[int(nb_canaux*0.5), :, :].numpy()
    B = image_tensor[int(nb_canaux*0.1), :, :].numpy()

image_rgb_visu = np.stack([R, G, B], axis=-1)
# Pour que l'image soit bien lumineuse à l'écran, on peut booster un peu le contraste
image_rgb_visu = np.clip(image_rgb_visu * 1.5, 0, 1)

# Format pour le réseau : (Batch, Canaux, Hauteur, Largeur)
image_tensor = image_tensor.unsqueeze(0)
print(f"Image prête pour le réseau ! Taille : {image_tensor.shape}")

# ------------------------------------------------------------------
# ÉTAPE 4 : PASSAGE DANS LE RÉSEAU
# ------------------------------------------------------------------
print("Extraction des caractéristiques (Alphas)...")
with torch.no_grad():
    _ = model.encode(image_tensor, img_id=None, sigmas=None, ssl_idx=None)

alpha_1 = alphas_capture['alpha_1'].squeeze(0) # (Nombre_atomes, H, W)
nb_atomes, h, w = alpha_1.shape

# ------------------------------------------------------------------
# ÉTAPE 5 : CLUSTERING K-MEANS
# ------------------------------------------------------------------
print("Lancement du clustering...")
alpha_features = alpha_1.permute(1, 2, 0).numpy()
pixels_flat = alpha_features.reshape(h * w, nb_atomes)

n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=2)
labels_flat = kmeans.fit_predict(pixels_flat)
segmentation_map = labels_flat.reshape(h, w)

# ------------------------------------------------------------------
# ÉTAPE 6 : VISUALISATION COMPLÈTE
# ------------------------------------------------------------------
plt.figure(figsize=(12, 6))

# Affichage de l'image originale (Pseudo-RGB)
plt.subplot(1, 2, 1)
plt.imshow(image_rgb_visu)
plt.title("Image Originale (Pseudo-RGB : Bandes 28, 15, 9)")
plt.axis('off')

# Affichage de la carte de clustering
plt.subplot(1, 2, 2)
plt.imshow(segmentation_map, cmap='Set1')
plt.title(f"Segmentation via les Alphas ({n_clusters} clusters)")
plt.colorbar(label="Numéro du cluster (Matériau)")
plt.axis('off')

plt.tight_layout()
plt.show()