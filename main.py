"""
main.py — AI Prompt Analysis Pipeline (Prompt-Driven Edition)
Alur: USER_PROMPT (env var) → gemini-1.5-flash (Analisis) → Groq LLaMA3-70B (Humanize) → Simpan .md
"""

import os
import sys
import logging
import datetime
import requests

# ─────────────────────────────────────────────
# KONFIGURASI LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

TODAY     = datetime.date.today().isoformat()
NOW_UTC   = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")


# ─────────────────────────────────────────────
# ENVIRONMENT VARIABLES
# ─────────────────────────────────────────────
def get_env(key: str, required: bool = True) -> str:
    val = os.environ.get(key, "").strip()
    if required and not val:
        log.error(f"Environment variable '{key}' tidak ditemukan atau kosong!")
        log.error(
            "Pastikan secret sudah diisi di: "
            "GitHub Repo → Settings → Secrets and variables → Actions"
        )
        sys.exit(1)
    return val

GEMINI_API_KEY   = get_env("GEMINI_API_KEY")
GROQ_API_KEY     = get_env("GROQ_API_KEY")
HUMANIZE_API_KEY = get_env("HUMANIZE_API_KEY", required=False)  # Opsional
GITHUB_TOKEN     = get_env("GITHUB_TOKEN",      required=False)  # Opsional

# Prompt dari input manual GitHub Actions.
# Jika kosong (trigger dari schedule/cron), gunakan prompt default.
DEFAULT_PROMPT = (
    "Analisis tren terbaru dalam dunia pengembangan software dan AI pada tahun 2025. "
    "Fokus pada teknologi yang paling banyak diadopsi developer, tantangan utama yang dihadapi, "
    "dan prediksi perkembangan ke depan."
)
USER_PROMPT = get_env("USER_PROMPT", required=False) or DEFAULT_PROMPT


# ─────────────────────────────────────────────
# INISIALISASI SDK GEMINI
# ─────────────────────────────────────────────
try:
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    log.info("Gemini SDK berhasil diinisialisasi (model: gemini-1.5-flash).")
except ImportError:
    log.error("Library 'google-generativeai' tidak terinstall.")
    sys.exit(1)


# ══════════════════════════════════════════════
# LANGKAH 1 — ANALISIS MENDALAM DENGAN GEMINI
# ══════════════════════════════════════════════
GEMINI_SYSTEM_PROMPT = """\
Kamu adalah analis teknologi dan riset senior yang berpengalaman.
Tugasmu adalah menghasilkan analisis mendalam, terstruktur, dan berbasis data
atas topik atau pertanyaan yang diberikan pengguna.
Selalu tulis dalam Bahasa Indonesia yang profesional namun mudah dipahami.
"""

GEMINI_USER_PROMPT_TEMPLATE = """\
Tanggal analisis: {date}

Topik / Pertanyaan dari pengguna:
\"\"\"{user_prompt}\"\"\"

Instruksi:
1. Berikan analisis mendalam dan komprehensif tentang topik di atas.
2. Gunakan struktur yang jelas: latar belakang, temuan utama, implikasi, dan kesimpulan.
3. Sertakan poin-poin kunci dalam format bullet point di setiap bagian.
4. Panjang: 400–600 kata.
5. Tulis dalam Bahasa Indonesia.
"""


def analyze_with_gemini(user_prompt: str) -> str:
    """
    Mengirimkan user_prompt ke gemini-1.5-flash untuk dianalisis secara mendalam.
    Mengembalikan teks hasil analisis, atau pesan error jika gagal.
    """
    log.info(f"Mengirim prompt ke gemini-1.5-flash...")
    log.info(f"Prompt preview: \"{user_prompt[:80]}...\"")

    full_prompt = GEMINI_USER_PROMPT_TEMPLATE.format(date=TODAY, user_prompt=user_prompt)

    try:
        response = gemini_model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=1024,
                temperature=0.7,
            ),
            # System instruction dimasukkan sebagai bagian dari model config
        )
        result = response.text.strip()
        log.info("Gemini analisis selesai.")
        return result
    except Exception as e:
        log.error(f"Gemini error: {e}")
        return f"[Gemini GAGAL: {type(e).__name__} — {e}]"


# ══════════════════════════════════════════════
# LANGKAH 2 — GROQ / LLAMA3 SEBAGAI HUMANIZER
# ══════════════════════════════════════════════
GROQ_SYSTEM_PROMPT = """\
Kamu adalah seorang penulis konten teknologi profesional berbahasa Indonesia.
Tugas utamamu adalah mengubah teks analisis AI menjadi tulisan yang terasa
natural, mengalir, dan enak dibaca oleh manusia — bukan seperti tulisan robot.
Pertahankan semua fakta dan informasi penting, tetapi perbaiki gaya bahasanya.
"""

GROQ_USER_PROMPT_TEMPLATE = """\
Berikut adalah draft analisis yang perlu kamu "humanize" (buat lebih natural):

─────────────────────────────────────
{gemini_result}
─────────────────────────────────────

Instruksimu:
1. Pertahankan semua informasi dan fakta dari draft di atas — JANGAN ada yang dihapus.
2. Perbaiki gaya bahasa agar terasa lebih natural, hangat, dan mudah dipahami manusia.
3. Hilangkan frasa klise khas AI seperti "Tentu saja", "Sebagai kesimpulan", "Penting untuk dicatat", dll.
4. Pertahankan format Markdown (heading, bullet point, bold).
5. Jangan tambahkan kalimat pembuka seperti "Berikut versi yang sudah dihumanize..." — langsung tulis kontennya.
6. Panjang akhir: setara dengan draft asli (tidak perlu diperpendek).
"""


def humanize_with_groq(gemini_result: str) -> str:
    """
    Mengirimkan hasil analisis Gemini ke Groq (LLaMA3-70B) untuk di-humanize.
    Mengembalikan teks yang sudah diperhalus, atau fallback jika gagal.
    """
    log.info("Mengirim hasil ke Groq LLaMA3-70B untuk di-humanize...")

    # Jika Gemini gagal, tidak ada gunanya memanggil Groq
    if gemini_result.startswith("[Gemini GAGAL"):
        log.warning("Gemini gagal, melewati langkah Groq.")
        return (
            "## Analisis Tidak Tersedia\n\n"
            "_Proses analisis gagal karena `GEMINI_API_KEY` tidak valid atau habis kuotanya. "
            "Periksa kembali secret di GitHub Actions._"
        )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}",
    }
    payload = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": GROQ_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": GROQ_USER_PROMPT_TEMPLATE.format(gemini_result=gemini_result),
            },
        ],
        "max_tokens": 1500,
        "temperature": 0.6,
    }

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        result = data["choices"][0]["message"]["content"].strip()
        log.info("Groq humanize selesai.")
        return result

    except requests.RequestException as e:
        log.error(f"Groq API error: {e}")
        log.warning("Menggunakan hasil Gemini mentah sebagai fallback.")
        return (
            "> *Catatan: Proses humanize oleh Groq tidak tersedia. "
            "Berikut hasil analisis langsung dari Gemini.*\n\n"
            + gemini_result
        )
    except (KeyError, IndexError) as e:
        log.error(f"Groq response parsing error: {e}")
        return gemini_result


# ══════════════════════════════════════════════
# LANGKAH 3 — KIRIM KE HUMANIZE AI (HTTP POST)
# ══════════════════════════════════════════════
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  PETUNJUK KONFIGURASI HUMANIZE AI                                       │
# │                                                                         │
# │  Ganti nilai HUMANIZE_API_ENDPOINT dengan URL endpoint layanan Anda:   │
# │     "https://api.humanizeai.pro/v1/humanize"                            │
# │     "https://app.undetectable.ai/api/submit"                            │
# │                                                                         │
# │  Sesuaikan key di dalam `payload` sesuai dokumentasi API layanan.       │
# └─────────────────────────────────────────────────────────────────────────┘
HUMANIZE_API_ENDPOINT = "https://YOUR-HUMANIZE-AI-ENDPOINT-HERE/v1/humanize"  # ← GANTI INI


def post_to_humanize_service(text: str) -> str:
    """
    Mengirim teks final ke layanan Humanize AI eksternal via HTTP POST.
    Dilewati otomatis jika endpoint atau API key belum dikonfigurasi.
    """
    if "YOUR-HUMANIZE-AI-ENDPOINT-HERE" in HUMANIZE_API_ENDPOINT:
        log.warning("HUMANIZE_API_ENDPOINT belum dikonfigurasi. Melewati langkah ini.")
        return text

    if not HUMANIZE_API_KEY:
        log.warning("HUMANIZE_API_KEY kosong. Melewati langkah humanize.")
        return text

    log.info("Mengirim teks ke layanan Humanize AI eksternal...")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {HUMANIZE_API_KEY}",  # Ganti skema auth jika perlu
    }
    payload = {
        "content": text,          # Ganti key sesuai dokumentasi: "input", "text", dll.
        # "readability": "University",
        # "purpose": "General Writing",
    }
    try:
        response = requests.post(
            HUMANIZE_API_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        result = (
            data.get("output")
            or data.get("result")
            or data.get("humanized")
            or text
        )
        log.info("Layanan Humanize AI eksternal selesai.")
        return result

    except requests.RequestException as e:
        log.warning(f"Humanize AI eksternal gagal ({e}). Menggunakan teks sebelumnya.")
        return text


# ══════════════════════════════════════════════
# LANGKAH 4 — SIMPAN KE FILE MARKDOWN
# ══════════════════════════════════════════════
def save_to_markdown(content: str, user_prompt: str) -> str:
    """Menyimpan hasil akhir ke file Markdown bertanggal."""
    # Gunakan timestamp agar tidak terjadi konflik jika dijalankan beberapa kali
    timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d_%H%M")
    filename = f"hasil_analisis_{timestamp}.md"

    markdown = f"""# Laporan Analisis AI
**Tanggal & Waktu:** {NOW_UTC}
**Topik / Prompt:**
> {user_prompt}

---

{content}

---
*Laporan ini dibuat secara otomatis oleh GitHub Actions.*
*Pipeline: User Prompt → gemini-1.5-flash (Analisis) → Groq LLaMA3-70B (Humanize) → Humanize AI Service*
"""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(markdown)
    log.info(f"Hasil disimpan ke: {filename}")
    return filename


# ══════════════════════════════════════════════
# MAIN ENTRYPOINT
# ══════════════════════════════════════════════
def main():
    log.info("═══ Memulai pipeline analisis AI (Prompt-Driven Edition) ═══")
    log.info(f"Prompt diterima: \"{USER_PROMPT[:100]}...\"" if len(USER_PROMPT) > 100 else f"Prompt: \"{USER_PROMPT}\"")

    # 1. Analisis mendalam dengan gemini-1.5-flash
    gemini_result = analyze_with_gemini(USER_PROMPT)

    # 2. Humanize hasil Gemini menggunakan Groq LLaMA3-70B
    humanized_result = humanize_with_groq(gemini_result)

    # 3. Kirim ke layanan Humanize AI eksternal (opsional)
    final_text = post_to_humanize_service(humanized_result)

    # 4. Simpan ke file Markdown
    filename = save_to_markdown(final_text, USER_PROMPT)

    log.info(f"═══ Pipeline selesai. File tersimpan: {filename} ═══")


if __name__ == "__main__":
    main()
