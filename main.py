"""
main.py — GitHub Trending Repo Analyzer (Free Tier Edition)
Alur: GitHub API → Google Gemini (Analisis) → Groq/LLaMA (Juri) → Humanize AI → Simpan .md
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

TODAY = datetime.date.today().isoformat()  # Format: YYYY-MM-DD


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


# ─────────────────────────────────────────────
# INISIALISASI SDK GEMINI
# ─────────────────────────────────────────────
try:
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-1.5-pro")
    log.info("Gemini SDK berhasil diinisialisasi (model: gemini-1.5-pro).")
except ImportError:
    log.error("Library 'google-generativeai' tidak terinstall.")
    sys.exit(1)


# ══════════════════════════════════════════════
# LANGKAH 1 — AMBIL DATA DARI GITHUB REST API
# ══════════════════════════════════════════════
def fetch_trending_python_repos(top_n: int = 5) -> list[dict]:
    """
    Mengambil repo Python terpopuler (7 hari terakhir) via GitHub Search API.
    Fallback ke data dummy jika API tidak tersedia.
    """
    log.info("Mengambil data trending repo Python dari GitHub API...")

    week_ago = (datetime.date.today() - datetime.timedelta(days=7)).isoformat()
    url = "https://api.github.com/search/repositories"
    params = {
        "q": f"language:python created:>{week_ago}",
        "sort": "stars",
        "order": "desc",
        "per_page": top_n,
    }
    headers = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"

    try:
        response = requests.get(url, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        items = response.json().get("items", [])

        if not items:
            log.warning("GitHub API tidak mengembalikan repo. Menggunakan data dummy.")
            return _dummy_repos()

        repos = [
            {
                "name":        repo["full_name"],
                "description": repo.get("description") or "Tidak ada deskripsi.",
                "stars":       repo["stargazers_count"],
                "url":         repo["html_url"],
                "topics":      repo.get("topics", []),
                "language":    repo.get("language", "Python"),
            }
            for repo in items
        ]
        log.info(f"Berhasil mengambil {len(repos)} repo.")
        return repos

    except requests.RequestException as e:
        log.warning(f"GitHub API gagal ({e}). Menggunakan data dummy.")
        return _dummy_repos()


def _dummy_repos() -> list[dict]:
    """Fallback data jika GitHub API tidak dapat diakses."""
    return [
        {
            "name": "openai/openai-python",
            "description": "Official Python SDK for the OpenAI API",
            "stars": 25000,
            "url": "https://github.com/openai/openai-python",
            "topics": ["openai", "ai", "python"],
            "language": "Python",
        },
        {
            "name": "google/generative-ai-python",
            "description": "Google Generative AI Python SDK",
            "stars": 10000,
            "url": "https://github.com/google/generative-ai-python",
            "topics": ["gemini", "google", "ai"],
            "language": "Python",
        },
    ]


def format_repos_for_prompt(repos: list[dict]) -> str:
    """Mengubah list repo menjadi teks terformat untuk prompt AI."""
    lines = []
    for i, r in enumerate(repos, 1):
        lines.append(
            f"{i}. **{r['name']}** — ⭐ {r['stars']} bintang\n"
            f"   Deskripsi : {r['description']}\n"
            f"   Topik     : {', '.join(r['topics']) or 'N/A'}\n"
            f"   URL       : {r['url']}"
        )
    return "\n\n".join(lines)


# ══════════════════════════════════════════════
# LANGKAH 2 — ANALISIS DENGAN GOOGLE GEMINI
# ══════════════════════════════════════════════
ANALYSIS_PROMPT_TEMPLATE = """\
Berikut adalah repositori Python yang sedang trending minggu ini ({date}):

{repo_text}

Tugasmu:
1. Analisis tren dan tema umum dari repo-repo ini secara mendalam.
2. Identifikasi teknologi, pola arsitektur, atau kebutuhan pasar yang dicerminkan.
3. Berikan ringkasan yang informatif, terstruktur, dan mudah dipahami (maks. 400 kata).
4. Sertakan poin-poin utama dalam format bullet point.

Tuliskan hasilmu dalam Bahasa Indonesia.
"""


def analyze_with_gemini(repo_text: str) -> str:
    """
    Memanggil Google Gemini 1.5 Pro untuk menganalisis repo trending.
    Mengembalikan string hasil analisis, atau pesan error jika gagal.
    """
    log.info("Memanggil Google Gemini 1.5 Pro untuk analisis...")
    prompt = ANALYSIS_PROMPT_TEMPLATE.format(date=TODAY, repo_text=repo_text)
    try:
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=800,
                temperature=0.7,
            ),
        )
        result = response.text.strip()
        log.info("Gemini analisis selesai.")
        return result
    except Exception as e:
        log.error(f"Gemini error: {e}")
        return f"[Gemini GAGAL: {type(e).__name__} — {e}]"


# ══════════════════════════════════════════════
# LANGKAH 3 — GROQ / LLAMA SEBAGAI JURI
# ══════════════════════════════════════════════
JURY_PROMPT_TEMPLATE = """\
Kamu adalah Editor Senior sekaligus Juri teknologi yang bertugas menyempurnakan
sebuah laporan analisis tentang repositori Python trending minggu ini ({date}).

Berikut adalah draft analisis dari Google Gemini:

─────────────────────────────────────
{gemini_result}
─────────────────────────────────────

Tugasmu:
1. Evaluasi kualitas analisis di atas: apakah sudah komprehensif, akurat, dan mudah dipahami?
2. Perkuat argumen yang lemah, tambahkan sudut pandang yang terlewat, dan perbaiki alur penulisan.
3. Hasilkan satu laporan akhir yang lebih tajam, terstruktur, dan profesional.
4. Gunakan format Markdown lengkap: heading (##, ###), bullet points, dan bold untuk poin kunci.
5. Panjang laporan akhir: 500–700 kata dalam Bahasa Indonesia.

Catatan: Jika draft mengandung error atau tidak tersedia, buatlah analisis baru berdasarkan
konteks repo Python trending secara umum pada {date}.

Laporan akhir harus berdiri sendiri (STANDALONE) — langsung dimulai dengan konten, tanpa kalimat pembuka seperti "Berikut laporan..." atau "Tentu saja...".
"""


def synthesize_with_groq(gemini_result: str) -> str:
    """
    Menggunakan Groq API (LLaMA 3 70B) sebagai juri untuk menyempurnakan
    hasil analisis Gemini. Mengembalikan teks final atau fallback jika gagal.
    """
    log.info("Mengirim hasil ke Groq (LLaMA3-70B) untuk sintesis akhir...")

    # Jika Gemini gagal total, tetap lanjutkan dengan prompt yang sudah ada
    jury_prompt = JURY_PROMPT_TEMPLATE.format(
        date=TODAY,
        gemini_result=gemini_result,
    )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}",
    }
    payload = {
        "model": "llama3-70b-8192",
        "messages": [
            {
                "role": "system",
                "content": (
                    "Kamu adalah editor teknologi senior berbahasa Indonesia. "
                    "Tugas utamamu adalah menghasilkan laporan analisis teknologi "
                    "yang tajam, komprehensif, dan mudah dipahami oleh pembaca umum."
                ),
            },
            {"role": "user", "content": jury_prompt},
        ],
        "max_tokens": 1024,
        "temperature": 0.5,
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
        log.info("Groq sintesis akhir selesai.")
        return result

    except requests.RequestException as e:
        log.error(f"Groq API error: {e}")
        log.warning("Menggunakan hasil Gemini mentah sebagai fallback.")
        return _build_fallback_summary(gemini_result)
    except (KeyError, IndexError) as e:
        log.error(f"Groq response parsing error: {e}")
        return _build_fallback_summary(gemini_result)


def _build_fallback_summary(gemini_result: str) -> str:
    """Fallback jika Groq gagal: kembalikan hasil Gemini dengan header sederhana."""
    if gemini_result.startswith("[Gemini GAGAL"):
        return (
            "## Laporan Analisis (Fallback)\n\n"
            "_Semua layanan AI gagal menghasilkan analisis. "
            "Pastikan `GEMINI_API_KEY` dan `GROQ_API_KEY` valid di GitHub Secrets._"
        )
    return (
        "## Laporan Analisis Repo Python Trending\n\n"
        "> *Catatan: Sintesis Groq tidak tersedia. Berikut hasil analisis langsung dari Gemini.*\n\n"
        + gemini_result
    )


# ══════════════════════════════════════════════
# LANGKAH 4 — KIRIM KE HUMANIZE AI (HTTP POST)
# ══════════════════════════════════════════════
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  PETUNJUK KONFIGURASI HUMANIZE AI                                       │
# │                                                                         │
# │  Ganti nilai HUMANIZE_API_ENDPOINT di bawah dengan URL endpoint         │
# │  resmi dari layanan Humanize AI yang Anda gunakan, contoh:              │
# │     "https://api.humanizeai.pro/v1/humanize"                            │
# │     "https://app.undetectable.ai/api/submit"                            │
# │                                                                         │
# │  Sesuaikan key di dalam `payload` sesuai dokumentasi API layanan:       │
# │  Beberapa API menggunakan "content", "input", atau "text".              │
# └─────────────────────────────────────────────────────────────────────────┘
HUMANIZE_API_ENDPOINT = "https://YOUR-HUMANIZE-AI-ENDPOINT-HERE/v1/humanize"  # ← GANTI INI


def humanize_text(text: str) -> str:
    """
    Mengirim teks final ke layanan Humanize AI via HTTP POST.
    Jika endpoint belum dikonfigurasi atau request gagal → kembalikan teks asli.
    """
    if "YOUR-HUMANIZE-AI-ENDPOINT-HERE" in HUMANIZE_API_ENDPOINT:
        log.warning("HUMANIZE_API_ENDPOINT belum dikonfigurasi. Melewati langkah ini.")
        return text

    if not HUMANIZE_API_KEY:
        log.warning("HUMANIZE_API_KEY kosong. Melewati langkah humanize.")
        return text

    log.info("Mengirim teks ke Humanize AI...")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {HUMANIZE_API_KEY}",  # Ganti skema auth jika perlu
    }
    # ▼ Sesuaikan struktur payload dengan dokumentasi API Anda
    payload = {
        "content": text,          # Beberapa API menggunakan "input" atau "text"
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
        humanized = (
            data.get("output")
            or data.get("result")
            or data.get("humanized")
            or text
        )
        log.info("Humanize AI selesai.")
        return humanized

    except requests.RequestException as e:
        log.warning(f"Humanize AI gagal ({e}). Menggunakan teks sintesis asli.")
        return text


# ══════════════════════════════════════════════
# LANGKAH 5 — SIMPAN KE FILE MARKDOWN
# ══════════════════════════════════════════════
def save_to_markdown(content: str, repos: list[dict]) -> str:
    """Menyimpan hasil akhir ke file Markdown bertanggal."""
    filename = f"hasil_analisis_{TODAY}.md"
    repo_links = "\n".join(
        f"- [{r['name']}]({r['url']}) — ⭐ {r['stars']}" for r in repos
    )
    markdown = f"""# Laporan Analisis Repo Python Trending
**Tanggal:** {TODAY}
**Sumber Data:** GitHub Search API (repo Python trending 7 hari terakhir)

---

## Repo yang Dianalisis

{repo_links}

---

{content}

---
*Laporan ini dibuat secara otomatis oleh GitHub Actions.*
*Pipeline: GitHub API → Gemini 1.5 Pro (Analisis) → Groq LLaMA3-70B (Juri) → Humanize AI*
"""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(markdown)
    log.info(f"Hasil disimpan ke: {filename}")
    return filename


# ══════════════════════════════════════════════
# MAIN ENTRYPOINT
# ══════════════════════════════════════════════
def main():
    log.info("═══ Memulai pipeline analisis AI (Free Tier Edition) ═══")

    # 1. Ambil data trending dari GitHub
    repos = fetch_trending_python_repos(top_n=5)
    repo_text = format_repos_for_prompt(repos)

    # 2. Analisis dengan Gemini 1.5 Pro
    gemini_result = analyze_with_gemini(repo_text)
    log.info(f"[Gemini] {gemini_result[:120].replace(chr(10), ' ')}...")

    # 3. Sintesis & penyempurnaan oleh Groq / LLaMA3-70B
    final_synthesis = synthesize_with_groq(gemini_result)

    # 4. Humanize teks final (opsional — lewati jika endpoint belum dikonfigurasi)
    humanized_text = humanize_text(final_synthesis)

    # 5. Simpan ke file Markdown
    filename = save_to_markdown(humanized_text, repos)

    log.info(f"═══ Pipeline selesai. File tersimpan: {filename} ═══")


if __name__ == "__main__":
    main()
