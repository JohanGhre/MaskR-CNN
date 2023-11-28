from pytube import YouTube


import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def telecharger_video(url, output_path):
    try:
        # Créer une instance de l'objet YouTube
        video = YouTube(url)

        # Sélectionner la plus haute résolution disponible
        stream = video.streams.get_highest_resolution()

        # Télécharger la vidéo en spécifiant le chemin de sortie
        stream.download(output_path)

        print("Téléchargement terminé.")
    except Exception as e:
        print("Une erreur est survenue lors du téléchargement :", str(e))

# URL de la vidéo YouTube à télécharger
url = ""

# Chemin de sortie du fichier téléchargé
output_path = ""

# Appeler la fonction de téléchargement
telecharger_video(url, output_path)
