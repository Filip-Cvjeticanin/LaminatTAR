import os
import time
from openai import OpenAI

# 1. Inicijalizacija OpenAI klijenta
MOJ_KLJUC = "kljucic"
client = OpenAI(api_key=MOJ_KLJUC)

# Postavke jezika i kategorija
JEZICI = ["French", "German", "Russian"]
KATEGORIJE = ["satire", "opinion", "reporting"]
BROJ_PRIMJERA_PO_JEZIKU = 200

# STVARANJE MAPE: Definiramo ime mape u koju sve spremamo
NAZIV_MAPE = "sinteticki_dataset_projekt"

# Ako mapa ne postoji na Desktopu (ili gdje vec pokreces skriptu), kôd je sam stvara
if not os.path.exists(NAZIV_MAPE):
    os.makedirs(NAZIV_MAPE)
    print(f"Stvorena je nova mapa: {NAZIV_MAPE}")

# Putanja do glavne datoteke unutar te nove mape
LABELS_OUTPUT_FILE = os.path.join(NAZIV_MAPE, "synthetic_train_labels.txt")

print("Zapocinjem generiranje REPLICIRANOG SINTETICKOG DATASETA za 3 jezika...")

# Početni ID za sintetičke artikle
trenutni_sinteticki_id = 999000000

for jezik in JEZICI:
    print(f"\n--- Pokrecem jezik: {jezik.upper()} ---")
    
    for i in range(BROJ_PRIMJERA_PO_JEZIKU):
        # Rotiramo kategorije (jedna trećina od svake)
        oznaka = KATEGORIJE[i % len(KATEGORIJE)]
        
        # Replikacija promptova iz znanstvenog rada
        if oznaka == "satire":
            prompt = f"Write a funny satirical article of at least 350 words in {jezik}."
        elif oznaka == "opinion":
            prompt = f"Write an opinion article of at least 350 words in {jezik}."
        elif oznaka == "reporting":
            prompt = f"Write a report article of at least 350 words in {jezik}."
            
        try:
            # Pozivamo model
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a professional news writer. Return ONLY the generated article text, without any introductory or concluding remarks, titles, or meta-text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.85
            )
            
            # Izvlačimo čisti tekst i cistimo nove redove
            generirani_tekst = response.choices[0].message.content.strip()
            generirani_tekst = generirani_tekst.replace("\n", " ")
            
            # Putanja do pojedinacnog clanka UNUTAR nove mape
            ime_artikla = f"synthetic_article{trenutni_sinteticki_id}.txt"
            putanja_artikla = os.path.join(NAZIV_MAPE, ime_artikla)
            
            # 1. Spremanje teksta u datoteku unutar mape
            with open(putanja_artikla, "w", encoding="utf-8") as art_f:
                art_f.write(generirani_tekst)
                
            # 2. Zapisivanje u glavnu datoteku unutar mape
            with open(LABELS_OUTPUT_FILE, "a", encoding="utf-8") as labels_f:
                labels_f.write(f"{trenutni_sinteticki_id}\t{oznaka}\n")
                
            print(f"Uspjeh | {jezik} | {i+1} od {BROJ_PRIMJERA_PO_JEZIKU} | ID: {trenutni_sinteticki_id}")
            
            trenutni_sinteticki_id += 1
            time.sleep(0.3)
            
        except:
            print(f"Greska na indeksu {i} za jezik {jezik}. Provjeri API kljuc ili internet vezu.")
            time.sleep(2)
            continue

print(f"\nSve je zavrseno! Cijeli dataset se nalazi u mapi: {NAZIV_MAPE}")
