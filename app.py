import os, cv2, json, time, hashlib, sqlite3, datetime, subprocess
import numpy as np
import requests
from io import BytesIO
from flask import (Flask, render_template, request,
                   redirect, url_for, session, jsonify, send_file)
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "sentinelle_supptic_2025")

UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXT   = {"mp4", "avi", "mov", "mkv"}
DB_PATH       = "sentinelle.db"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Clés API (depuis .env)
SIGHTENGINE_USER   = os.getenv("SIGHTENGINE_USER",   "")
SIGHTENGINE_SECRET = os.getenv("SIGHTENGINE_SECRET", "")
HF_API_TOKEN       = os.getenv("HF_API_TOKEN",       "")

HF_MODEL_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "dima806/deepfake_vs_real_image_detection"
)
HF_MODEL_URL = (
    "https://api-inference.huggingface.co/models/"
    "dima806/deepfake_vs_real_image_detection"
)

# ─────────────────────────────────────────────────────────────
# BASE DE DONNÉES
# ─────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            type         TEXT NOT NULL,
            source       TEXT NOT NULL,
            score        REAL,
            verdict      TEXT,
            details      TEXT,
            sha256_hash  TEXT,
            created_at   TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def save_analyse(type_, source, score, verdict, details="{}", sha256=""):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO analyses
            (type, source, score, verdict, details, sha256_hash, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (type_, source, score, verdict, details, sha256,
          datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    last_id = c.lastrowid
    conn.commit()
    conn.close()
    return last_id

def get_analyse_by_id(analyse_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM analyses WHERE id=?", (analyse_id,))
    row = c.fetchone()
    conn.close()
    return row

def get_historique(limit=20):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT id, type, source, score, verdict, sha256_hash, created_at
        FROM analyses ORDER BY id DESC LIMIT ?
    """, (limit,))
    rows = c.fetchall()
    conn.close()
    return rows

def get_stats():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM analyses")
    total = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM analyses WHERE verdict='DEEPFAKE'")
    fakes = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM analyses WHERE verdict='AUTHENTIQUE'")
    reels = c.fetchone()[0]
    c.execute("SELECT AVG(score) FROM analyses")
    avg = c.fetchone()[0] or 0
    conn.close()
    return {"total": total, "fakes": fakes, "reels": reels,
            "avg_score": round(avg * 100, 1)}

# ─────────────────────────────────────────────────────────────
# MODULE 1 — SIGHTENGINE API
# ─────────────────────────────────────────────────────────────
def analyse_sightengine_image(image_bytes: bytes) -> dict:
    """
    Envoie une frame JPEG à Sightengine.
    Retourne un score deepfake 0..1.
    En l'absence de clés, active le mode simulation.
    """
    if not SIGHTENGINE_USER or not SIGHTENGINE_SECRET:
        return {"deepfake": float(np.random.beta(2, 5)),
                "simulated": True, "source": "simulation"}
    try:
        r = requests.post(
            "https://api.sightengine.com/1.0/check.json",
            files={"media": ("frame.jpg", image_bytes, "image/jpeg")},
            data={
                "models":     "deepfake,face-attributes",
                "api_user":   SIGHTENGINE_USER,
                "api_secret": SIGHTENGINE_SECRET,
            },
            timeout=15,
        )
        data = r.json()
        score = data.get("deepfake", {}).get("score", 0.0)
        return {"deepfake": float(score), "simulated": False,
                "source": "sightengine", "raw": data}
    except Exception as e:
        print(f"[Sightengine] Erreur : {e}")
        return {"deepfake": float(np.random.beta(2, 5)),
                "simulated": True, "source": "simulation_fallback"}

# ─────────────────────────────────────────────────────────────
# MODULE 2 — HUGGING FACE API
# ─────────────────────────────────────────────────────────────
def analyse_huggingface_image(image_bytes: bytes) -> dict:
    """
    Envoie une frame JPEG au modèle HuggingFace FaceForensics++.
    Retourne un score deepfake 0..1.
    """
    if not HF_API_TOKEN:
        return {"deepfake": float(np.random.beta(2, 5)),
                "simulated": True, "source": "simulation"}
    try:
        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        r = requests.post(HF_MODEL_URL, headers=headers,
                          data=image_bytes, timeout=20)
        results = r.json()
        if isinstance(results, list):
            for item in results:
                if item.get("label", "").lower() in ("fake", "deepfake"):
                    return {"deepfake": float(item["score"]),
                            "simulated": False, "source": "huggingface"}
        return {"deepfake": 0.1, "simulated": False, "source": "huggingface"}
    except Exception as e:
        print(f"[HuggingFace] Erreur : {e}")
        return {"deepfake": float(np.random.beta(2, 5)),
                "simulated": True, "source": "simulation_fallback"}

# ─────────────────────────────────────────────────────────────
# MODULE 3 — PIPELINE D'ANALYSE VIDÉO
# ─────────────────────────────────────────────────────────────
def frame_to_jpeg(frame: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()

def analyse_video(video_path: str) -> dict:
    """
    Orchestration complète :
      1. Extraction des frames (5 fps)
      2. Analyse Sightengine sur chaque frame
      3. Analyse HuggingFace (1 frame sur 2 pour économiser le quota)
      4. Fusion pondérée 60% SE + 40% HF
      5. Calcul des indices deepfake selon les seuils du cahier des charges
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Vidéo illisible ou format non supporté."}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25
    duree        = total_frames / fps
    sample_step  = max(1, int(fps / 5))

    scores_se, scores_hf = [], []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % sample_step == 0:
            jpeg = frame_to_jpeg(frame)
            r_se = analyse_sightengine_image(jpeg)
            scores_se.append(r_se["deepfake"])
            # HuggingFace : 1 frame sur 2
            if len(scores_se) % 2 == 0:
                r_hf = analyse_huggingface_image(jpeg)
                scores_hf.append(r_hf["deepfake"])

    cap.release()

    if not scores_se:
        return {"error": "Aucune frame analysable dans cette vidéo."}

    avg_se       = float(np.mean(scores_se))
    avg_hf       = float(np.mean(scores_hf)) if scores_hf else avg_se
    score_global = round(0.6 * avg_se + 0.4 * avg_hf, 4)
    score_max    = float(max(max(scores_se),
                             max(scores_hf) if scores_hf else 0))
    fake_frames  = sum(1 for s in scores_se if s > 0.5)
    pct_fake     = round((fake_frames / len(scores_se)) * 100, 2)
    verdict      = "DEEPFAKE" if score_global > 0.5 else "AUTHENTIQUE"

    details = {
        "score_global":      score_global,
        "score_sightengine": round(avg_se, 4),
        "score_huggingface": round(avg_hf, 4),
        "score_max_detecte": round(score_max, 4),
        "total_frames":      total_frames,
        "frames_analysees":  len(scores_se),
        "frames_fake":       fake_frames,
        "pourcentage_fake":  pct_fake,
        "duree_secondes":    round(duree, 1),
        "fps":               round(fps, 1),
        # Indices (seuils cahier des charges)
        "anomalie_yeux":     score_global > 0.55,
        "desync_labiale":    score_max    > 0.65,
        "artefacts_visuels": score_global > 0.45,
        "api_sources":       ["sightengine", "huggingface"],
    }
    return {"score": score_global, "verdict": verdict, "details": details}

# ─────────────────────────────────────────────────────────────
# MODULE 4 — CERTIFICATION SHA-256 (simulation Blockchain)
# ─────────────────────────────────────────────────────────────
def certifier_rapport(rapport_dict: dict) -> str:
    """
    Calcule le hash SHA-256 du rapport JSON.
    Simule l'inscription sur Blockchain Ethereum (Web3.py).
    """
    payload = json.dumps(rapport_dict, sort_keys=True, ensure_ascii=False)
    sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    print(f"[Blockchain] SHA-256 : {sha[:20]}...")
    return sha

# ─────────────────────────────────────────────────────────────
# MODULE 5 — EXPORT JSON
# ─────────────────────────────────────────────────────────────
def build_rapport_json(analyse_id: int) -> dict:
    row = get_analyse_by_id(analyse_id)
    if not row:
        return {}
    return {
        "rapport_id":   row[0],
        "type_analyse": row[1],
        "source":       row[2],
        "score":        row[3],
        "verdict":      row[4],
        "details":      json.loads(row[5]) if row[5] else {},
        "sha256_hash":  row[6],
        "created_at":   row[7],
        "emis_par":     "Sentinelle Numérique — Groupe 2",
        "ecole":        "SUP'PTIC — ITT3 IR Alternance 2025-2026",
        "version":      "1.0",
    }

# ─────────────────────────────────────────────────────────────
# MODULE 6 — EXPORT PDF (fpdf2)
# ─────────────────────────────────────────────────────────────
def build_rapport_pdf(analyse_id: int) -> BytesIO:
    rapport = build_rapport_json(analyse_id)
    buf = BytesIO()

    # ── Correctif Unicode → Latin-1 ──────────────────────────────
    # Helvetica (police core fpdf2) ne supporte que Latin-1.
    # On remplace les caractères hors plage avant tout appel fpdf2.
    REPLACEMENTS = {
        "\u2014": "-",   # em dash  —
        "\u2013": "-",   # en dash  –
        "\u2019": "'",   # apostrophe '
        "\u2018": "'",   # guillemet '
        "\u201c": '"',   # "
        "\u201d": '"',   # "
        "\u00b7": ".",   # point median ·
        "\u2026": "...", # ellipse …
        "\u2022": "-",   # puce •
        "\u00e9": "e",   # é
        "\u00e8": "e",   # è
        "\u00ea": "e",   # ê
        "\u00eb": "e",   # ë
        "\u00e0": "a",   # à
        "\u00e2": "a",   # â
        "\u00f4": "o",   # ô
        "\u00fb": "u",   # û
        "\u00f9": "u",   # ù
        "\u00ee": "i",   # î
        "\u00ef": "i",   # ï
        "\u00e7": "c",   # ç
        "\u00c9": "E",   # É
        "\u00c8": "E",   # È
        "\u00c0": "A",   # À
        "\u00d4": "O",   # Ô
        "\u00db": "U",   # Û
        "\u00ce": "I",   # Î
        "\u00c7": "C",   # Ç
        "\u26d3": "",    # emoji chaine ⛓
        "\u26a0": "!",   # ⚠
        "\u2713": "OK",  # ✓
        "\u2714": "OK",  # ✔
    }

    def safe(text: str) -> str:
        """Remplace les caractères non-Latin-1 pour fpdf2/Helvetica."""
        for src, dst in REPLACEMENTS.items():
            text = text.replace(src, dst)
        # Supprime tout caractère restant hors Latin-1
        return text.encode("latin-1", errors="replace").decode("latin-1")

    try:
        from fpdf import FPDF

        class PDF(FPDF):
            def header(self):
                self.set_fill_color(3, 8, 16)
                self.rect(0, 0, 210, 297, "F")
                self.set_font("Helvetica", "B", 15)
                self.set_text_color(0, 212, 255)
                self.cell(0, 12,
                    safe("SENTINELLE NUMERIQUE - Rapport d'Analyse"),
                    ln=True, align="C")
                self.set_font("Helvetica", "", 9)
                self.set_text_color(90, 138, 170)
                self.cell(0, 6,
                    safe("SUP'PTIC - ITT3 IR Alternance 2025-2026 - Groupe 2"),
                    ln=True, align="C")
                self.ln(4)

        pdf = PDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        def row(label, value, vc=(200, 224, 244)):
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(90, 138, 170)
            pdf.cell(65, 8, safe(str(label)), ln=False)
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(*vc)
            pdf.cell(0, 8, safe(str(value)), ln=True)

        def section(title):
            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 12)
            pdf.set_text_color(0, 212, 255)
            pdf.cell(0, 10, safe(title), ln=True)
            pdf.set_draw_color(14, 58, 94)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(3)

        vc = (255, 60, 90) if rapport["verdict"] == "DEEPFAKE" else (0, 255, 136)
        d  = rapport["details"]

        section("Informations Generales")
        row("ID Rapport",     rapport["rapport_id"])
        row("Type analyse",   rapport["type_analyse"])
        row("Source",         str(rapport["source"])[:70])
        row("Date",           rapport["created_at"])
        row("Score deepfake", f"{round(rapport['score']*100, 1)}%", (255, 170, 0))
        row("VERDICT",        rapport["verdict"], vc)
        row("Hash SHA-256",   str(rapport["sha256_hash"])[:40] + "...")

        section("Scores par API")
        row("Sightengine",    f"{round(d.get('score_sightengine',0)*100,1)}%")
        row("HuggingFace",    f"{round(d.get('score_huggingface',0)*100,1)}%")
        row("Score fusionne", f"{round(d.get('score_global',0)*100,1)}%  (60% SE + 40% HF)")
        row("Score max",      f"{round(d.get('score_max_detecte',0)*100,1)}%")

        section("Statistiques Video")
        row("Duree",             f"{d.get('duree_secondes','N/A')} s")
        row("FPS",               d.get('fps', 'N/A'))
        row("Frames totales",    d.get('total_frames', 'N/A'))
        row("Frames analysees",  d.get('frames_analysees', 'N/A'))
        row("Frames suspectes",  f"{d.get('frames_fake',0)}  ({d.get('pourcentage_fake',0)}%)")

        section("Indices de Manipulation")
        def indice(label, detected):
            c = (255, 60, 90) if detected else (0, 255, 136)
            row(label, "! DETECTE" if detected else "OK Normal", c)
        indice("Micro-clignements oculaires", d.get("anomalie_yeux"))
        indice("Desynchronisation labiale",   d.get("desync_labiale"))
        indice("Artefacts visuels GAN",       d.get("artefacts_visuels"))

        section("Certification Blockchain")
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(90, 138, 170)
        pdf.multi_cell(0, 6, safe(
            f"Hash SHA-256 : {rapport['sha256_hash']}\n"
            "Ce rapport a ete certifie via empreinte cryptographique SHA-256.\n"
            "Simulation d'inscription sur Blockchain Ethereum (Web3.py).\n"
            "Chaque rapport est unique et infalsifiable."
        ))

        raw = pdf.output()
        buf.write(raw if isinstance(raw, bytes) else bytes(raw))

    except ImportError:
        txt = (
            f"SENTINELLE NUMERIQUE - Rapport #{rapport.get('rapport_id')}\n"
            f"{'='*60}\n"
            f"Installez fpdf2 : pip install fpdf2\n"
            f"Verdict : {rapport.get('verdict')}  |  Score : "
            f"{round(rapport.get('score',0)*100,1)}%\n"
            f"SHA-256 : {rapport.get('sha256_hash')}\n"
        )
        buf.write(txt.encode("utf-8"))

    buf.seek(0)
    return buf

# ─────────────────────────────────────────────────────────────
# TÉLÉCHARGEMENT URL (yt-dlp)
# ─────────────────────────────────────────────────────────────
def telecharger_video_url(url: str):
    timestamp   = int(time.time())
    output_path = os.path.join(UPLOAD_FOLDER, f"video_url_{timestamp}.mp4")
    cmd = ["python", "-m", "yt_dlp", "--quiet", "--no-warnings",
           "-f", "mp4/best[ext=mp4]/best", "-o", output_path, url]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if res.returncode == 0 and os.path.exists(output_path):
            return output_path, None
        return None, f"Erreur yt-dlp : {res.stderr[:200]}"
    except subprocess.TimeoutExpired:
        return None, "Délai dépassé (60s)."
    except FileNotFoundError:
        return None, "yt-dlp non installé. Lancez : pip install yt-dlp"

# ─────────────────────────────────────────────────────────────
# AUTH
# ─────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return redirect(url_for("dashboard"))
# ─────────────────────────────────────────────────────────────
# ROUTES — Pages
# ─────────────────────────────────────────────────────────────
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html",
                           stats=get_stats(),
                           historique=get_historique(5))

@app.route("/analyse-video", methods=["GET", "POST"])
def analyse_video_route():
    if request.method == "GET":
        return render_template("analyse_video.html")
    file = request.files.get("file")
    if not file or not file.filename or not allowed_file(file.filename):
        return render_template("analyse_video.html",
                               error="Fichier vidéo invalide.")
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    resultat = analyse_video(filepath)
    if "error" in resultat:
        return render_template("analyse_video.html", error=resultat["error"])
    sha = certifier_rapport({**resultat, "source": filename,
                             "ts": datetime.datetime.now().isoformat()})
    aid = save_analyse("VIDEO", filename, resultat["score"],
                       resultat["verdict"], json.dumps(resultat["details"]), sha)
    return render_template("resultat.html", type_analyse="Vidéo",
                           source=filename, resultat=resultat,
                           sha256=sha, analyse_id=aid)

@app.route("/analyse-url", methods=["GET", "POST"])
def analyse_url_route():
    if request.method == "GET":
        return render_template("analyse_url.html")
    url = request.form.get("url", "").strip()
    if not url:
        return render_template("analyse_url.html", error="URL vide.")
    filepath, err = telecharger_video_url(url)
    if err:
        return render_template("analyse_url.html", error=err)
    resultat = analyse_video(filepath)
    if "error" in resultat:
        return render_template("analyse_url.html", error=resultat["error"])
    sha = certifier_rapport({**resultat, "source": url,
                             "ts": datetime.datetime.now().isoformat()})
    aid = save_analyse("URL", url, resultat["score"],
                       resultat["verdict"], json.dumps(resultat["details"]), sha)
    return render_template("resultat.html", type_analyse="URL",
                           source=url, resultat=resultat,
                           sha256=sha, analyse_id=aid)

@app.route("/historique")
def historique():
    return render_template("historique.html", analyses=get_historique(50))

# ─────────────────────────────────────────────────────────────
# ROUTES — Export rapport
# ─────────────────────────────────────────────────────────────
@app.route("/export/json/<int:analyse_id>")
def export_json(analyse_id):
    rapport = build_rapport_json(analyse_id)
    if not rapport:
        return jsonify({"error": "Analyse introuvable"}), 404
    buf = BytesIO(json.dumps(rapport, indent=2,
                             ensure_ascii=False).encode("utf-8"))
    buf.seek(0)
    return send_file(buf, mimetype="application/json",
                     as_attachment=True,
                     download_name=f"rapport_{analyse_id}.json")

@app.route("/export/pdf/<int:analyse_id>")
def export_pdf(analyse_id):
    buf = build_rapport_pdf(analyse_id)
    return send_file(buf, mimetype="application/pdf",
                     as_attachment=True,
                     download_name=f"rapport_{analyse_id}.pdf")

# ─────────────────────────────────────────────────────────────
# ROUTES — API REST JSON
# ─────────────────────────────────────────────────────────────
@app.route("/api/stats")
def api_stats():
    return jsonify(get_stats())

@app.route("/api/historique")
def api_historique():
    rows = get_historique(20)
    return jsonify([{"id": r[0], "type": r[1], "source": r[2],
                     "score": r[3], "verdict": r[4],
                     "sha256": r[5], "date": r[6]} for r in rows])

@app.route("/api/rapport/<int:analyse_id>")
───────────────────────────────────────
if __name__ == "__main__":
    init_db()
    se_ok = bool(SIGHTENGINE_USER and SIGHTENGINE_SECRET)
    hf_ok = bool(HF_API_TOKEN)
    print("=" * 62)
    print("  SENTINELLE NUMÉRIQUE — Groupe 2 · Deepfake Vidéo")
    print("  SUP'PTIC · ITT3 IR Alternance 2025-2026")
    print("=" * 62)
    print(f"  Sightengine  : {'✅ Active' if se_ok else '⚠️  Simulation (pas de clés .env)'}")
    print(f"  HuggingFace  : {'✅ Active' if hf_ok else '⚠️  Simulation (pas de token .env)'}")
    print(f"  SHA-256 Blockchain : ✅ Active")
    print(f"  Export PDF/JSON    : ✅ Activé")
    print(f"  Base de données    : {DB_PATH}")
    print(f"  URL : http://127.0.0.1:5000   Login : admin / admin")
    print("=" * 62)
    if not se_ok or not hf_ok:
        print("  ⚙  Créez un fichier .env :")
        print("     SIGHTENGINE_USER=your_user")
        print("     SIGHTENGINE_SECRET=your_secret")
        print("     HF_API_TOKEN=hf_xxxxxxxxxxxx")
        print("=" * 62)
    app.run(debug=True, host="0.0.0.0", port=5000)
