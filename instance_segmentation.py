import cv2
import torch
import numpy as np
from torchvision import transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn
import random


def instance_segmentation(image):
    # Charger le modèle pré-entraîné Mask R-CNN
    model = maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    # Transformer l'image pour l'entrée du modèle
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    img_tensor = transform(image)
    img_tensor = img_tensor.unsqueeze(0)

    # Prédictions
    with torch.no_grad():
        predictions = model(img_tensor)

    # Vérifier si la clé 'masks' est présente dans les prédictions
    if 'masks' not in predictions[0]:
        print("Erreur : le modèle ne génère pas de masques dans ses prédictions.")
        return None

    # Créer une image noire de la même taille que l'image originale
    black_image = np.zeros_like(image)

    # Dessiner les masques des objets avec des couleurs aléatoires et ajouter des encadrements
    for i in range(predictions[0]['masks'].shape[0]):
        mask = predictions[0]['masks'][i, 0].mul(255).byte().cpu().numpy()
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        black_image[mask > 128] = color

        # Dessiner l'encadrement
        bbox = predictions[0]['boxes'][i].cpu().numpy()
        x1, y1, x2, y2 = bbox.astype(int)
        cv2.rectangle(black_image, (x1, y1), (x2, y2), color, 1)

    return black_image


def main():
    # Charger l'image
    image_path = " " # mettre le chemin vers l'image
    image = cv2.imread(image_path)

    if image is None:
        print("Erreur : impossible de charger l'image. Vérifiez le chemin du fichier et l'intégrité du fichier.")
        return

    # Appliquer la segmentation d'instances
    segmented_image = instance_segmentation(image)

    if segmented_image is None:
        print("Erreur : la segmentation d'instances a échoué.")
        return

    # Afficher les images
    cv2.imshow('Image originale', image)
    cv2.imshow('Image segmentée', segmented_image)

    # Attendre une touche pour fermer les fenêtres
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
