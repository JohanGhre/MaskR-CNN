import cv2
import torch
from torchvision import transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn
import random
import time


def instance_segmentation(image, model):
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

    # Dessiner les masques des objets avec des couleurs aléatoires et ajouter des encadrements
    for i in range(predictions[0]['masks'].shape[0]):
        mask = predictions[0]['masks'][i, 0].mul(255).byte().cpu().numpy()
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        image[mask > 128] = color

        # Dessiner l'encadrement
        bbox = predictions[0]['boxes'][i].cpu().numpy()
        x1, y1, x2, y2 = bbox.astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)

    return image


def main():
    # Ouvrir la vidéo
    video_path = ''  # Mettez le chemin vers votre vidéo
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print("Erreur : impossible d'ouvrir la vidéo. Vérifiez le chemin du fichier et l'intégrité du fichier.")
        return

    # Lire la première image de la vidéo pour obtenir les dimensions
    ret, frame = video.read()

    if not ret:
        print("Erreur : impossible de lire la première image de la vidéo.")
        video.release()
        return

    # Déduire le chemin de sortie en utilisant le nom du fichier vidéo d'entrée
    output_path = '/Users/johanghre/Documents/object_detection_maskrCNN /output_video.mp4'

    # Créer un objet VideoWriter pour écrire la vidéo segmentée
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video = cv2.VideoWriter(output_path, codec, fps, (frame_width, frame_height))

    # Charger le modèle pré-entraîné Mask R-CNN
    model = maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    # Segmenter chaque image de la vidéo et l'afficher
    while ret:
        # Début du temps de traitement
        start_time = time.time()

        # Réduire la résolution de l'image pour accélérer la détection
        resized_frame = cv2.resize(frame, (320, 240))

        # Appliquer la segmentation d'instances sur l'image
        segmented_image = instance_segmentation(resized_frame, model)

        if segmented_image is not None:
            # Redimensionner l'image segmentée à la taille d'origine
            segmented_image = cv2.resize(segmented_image, (frame_width, frame_height))

            # Afficher l'image segmentée
            cv2.imshow('Object Detection', segmented_image)
            cv2.waitKey(1)

            # Écrire l'image originale avec les objets détectés dans la vidéo de sortie
            output_video.write(segmented_image)

        # Lire l'image suivante de la vidéo
        ret, frame = video.read()

        # Temps de traitement de l'image
        processing_time = time.time() - start_time

        # Calculer le délai pour correspondre à la vitesse de la vidéo
        delay = max(1, int((1 / fps) - processing_time) * 1000)

        # Attendre pour synchroniser la vitesse de détection avec la vitesse de la vidéo
        cv2.waitKey(delay)

    # Libérer les ressources
    video.release()
    output_video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
