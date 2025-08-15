# Importiamo le librerie necessarie
import os
import torch
from PIL import Image
import torchvision.transforms as T


destinazione = 'SinteticData'
os.makedirs(destinazione, exist_ok=True)
cartelle_classi = [
    'actinic keratosis', 'basal cell carcinoma', 'melanoma', 
    'nevus', 'seborrheic keratosis', 'squamous cell carcinoma'
]


h_flip = T.RandomHorizontalFlip(p=1)
v_flip = T.RandomVerticalFlip(p=1)
rot = T.RandomRotation(degrees=90)
jit = T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.1)

for cartella in cartelle_classi:
    nuova_cartella = os.path.join(destinazione, cartella)
    os.makedirs(nuova_cartella, exist_ok=True)
    print(f"\nProcessando la cartella: {cartella}")
    i=0
    for img in os.listdir(cartella):
        file_originale = os.path.join(cartella, img)
        if os.path.isfile(file_originale) and img.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Copia 0: copia originale
            img_originale = Image.open(file_originale).convert('RGB')
            nome, estensione = os.path.splitext(img)
            nuovo_nome = f"{nome}_c0{estensione}"
            percorso = os.path.join(nuova_cartella, nuovo_nome)
            img_originale.save(percorso)
            if cartella==cartelle_classi[0] or cartella==cartelle_classi[5]:
                # Copia 1: flip orizzontale
                copia1 = h_flip(img_originale)
                nuovo_nome = f"{nome}_c1{estensione}"
                percorso = os.path.join(nuova_cartella, nuovo_nome)
                copia1.save(percorso)
                # Copia 2: flip verticale
                copia2 = v_flip(img_originale)
                nuovo_nome = f"{nome}_c2{estensione}"
                percorso = os.path.join(nuova_cartella, nuovo_nome)
                copia2.save(percorso)
                # Copia 3: entrambi i flip
                copia3 = h_flip(img_originale)
                copia3 = v_flip(copia3)
                nuovo_nome = f"{nome}_c3{estensione}"
                percorso = os.path.join(nuova_cartella, nuovo_nome)
                copia3.save(percorso)
                # Copia 4: Rotazione casuale
                copia4 = rot(img_originale)
                nuovo_nome = f"{nome}_c4{estensione}"
                percorso = os.path.join(nuova_cartella, nuovo_nome)
                copia4.save(percorso)
                # Copia 5: Jitter colore/contrasto
                copia5 = jit(img_originale)
                nuovo_nome = f"{nome}_c5{estensione}"
                percorso = os.path.join(nuova_cartella, nuovo_nome)
                copia5.save(percorso)
                
            elif cartella==cartelle_classi[4]:
                # Copia 1: flip orizzontale
                copia1 = h_flip(img_originale)
                nuovo_nome = f"{nome}_c1{estensione}"
                percorso = os.path.join(nuova_cartella, nuovo_nome)
                copia1.save(percorso)
                # Copia 2: flip verticale
                copia2 = v_flip(img_originale)
                nuovo_nome = f"{nome}_c2{estensione}"
                percorso = os.path.join(nuova_cartella, nuovo_nome)
                copia2.save(percorso)
                # Copia 3: Rotazione casuale
                rot = T.RandomRotation(degrees=90)
                copia3 = rot(img_originale)
                nuovo_nome = f"{nome}_c3{estensione}"
                percorso = os.path.join(nuova_cartella, nuovo_nome)
                copia3.save(percorso)
                # Copia 4: Jitter colore/contrasto
                jit = T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.1)
                copia4 = jit(img_originale)
                nuovo_nome = f"{nome}_c4{estensione}"
                percorso = os.path.join(nuova_cartella, nuovo_nome)
                copia4.save(percorso)
                
            elif cartella==cartelle_classi[1]:
                if i<1000:
                    copia1 = rot(img_originale)
                    nuovo_nome = f"{nome}_c1{estensione}"
                    percorso = os.path.join(nuova_cartella, nuovo_nome)
                    copia1.save(percorso)
                    i+=1
