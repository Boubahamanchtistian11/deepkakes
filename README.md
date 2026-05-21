# SENTINELLE NUMÉRIQUE — Groupe 2
## Spécialiste Deepfake Vidéo
**SUP'PTIC · ITT3 IR Alternance 2025-2026**

Membres : AWONO NGAH Arnold · KOAGNE DANDA Adrien · BOUBA HAMAN Christian  
Encadrant : M. SASSA Therance

---

## Structure du Projet

```
sentinelle/
├── app.py                  ← Application Flask + APIs + SHA-256 + Export
├── requirements.txt        ← Dépendances Python
├── .env.example            ← Template de configuration des clés API
├── sentinelle.db           ← Base SQLite (créée au lancement)
├── static/
│   └── uploads/            ← Vidéos téléchargées
└── templates/
    ├── base.html           ← Layout global (nav, styles)
    ├── login.html          ← Page de connexion animée
    ├── dashboard.html      ← Tableau de bord + statistiques
    ├── analyse_video.html  ← Upload vidéo locale (drag & drop)
    ├── analyse_url.html    ← Analyse par URL (TikTok/Reels/YouTube)
    ├── resultat.html       ← Rapport + export PDF/JSON + SHA-256
    └── historique.html     ← Historique filtrable + exports
```

---

## Installation

```bash
# 1. Créer un environnement virtuel
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/Mac

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Configurer les clés API
cp .env.example .env
# Éditez .env avec vos clés Sightengine et HuggingFace

# 4. Lancer l'application
python app.py
```

Accéder à : **http://127.0.0.1:5000**  
Login : `admin` / `admin`

---

## Configuration des APIs (.env)

```env
SIGHTENGINE_USER=your_user          # https://sightengine.com (2000 req/mois gratuit)
SIGHTENGINE_SECRET=your_secret
HF_API_TOKEN=hf_xxxxxxxxxxxx        # https://huggingface.co (1000 req/jour gratuit)
SECRET_KEY=votre_cle_secrete
```

> Sans clés → le projet fonctionne en **mode simulation** (scores aléatoires réalistes).

---

## Fonctionnalités (Cahier des Charges Livrable 1)

| Fonctionnalité | Statut | Description |
|---|---|---|
| Détection deepfake IA | ✅ | Sightengine API (2000 req/mois gratuit) |
| Modèle FaceForensics++ | ✅ | HuggingFace API (1000 req/jour gratuit) |
| Fusion des scores | ✅ | 60% Sightengine + 40% HuggingFace |
| Micro-clignements oculaires | ✅ | Score > 55% |
| Désynchronisation labiale | ✅ | Score max > 65% |
| Artefacts visuels GAN | ✅ | Score > 45% |
| Certification SHA-256 | ✅ | Simulation Blockchain Ethereum |
| Export JSON | ✅ | Rapport structuré téléchargeable |
| Export PDF | ✅ | Rapport mis en forme (fpdf2) |
| Analyse par URL | ✅ | TikTok, YouTube Shorts, Instagram Reels |
| Historique SQLite | ✅ | Filtrable, avec SHA-256 par ligne |
| Dashboard statistiques | ✅ | Total, deepfakes, authentiques, score moyen |

---

## APIs REST disponibles

| Endpoint | Description |
|---|---|
| `GET /api/stats` | Statistiques globales JSON |
| `GET /api/historique` | 20 dernières analyses JSON |
| `GET /api/rapport/<id>` | Rapport complet d'une analyse |
| `GET /export/json/<id>` | Télécharger rapport JSON |
| `GET /export/pdf/<id>` | Télécharger rapport PDF |

---

## Membres du Groupe 2

| Nom | Matricule |
|---|---|
| AWONO NGAH Arnold | 23T080 |
| KOAGNE DANDA Adrien | 23T037 |
| BOUBA HAMAN Christian | 23T014 |
"# deepkakes" 
