"""
main.py — GitHub Trending Repo Analyzer
Alur: GitHub API → 3 AI (Paralel) → OpenAI Juri → Humanize AI → Simpan .md
"""

import os
import json
import logging
import datetime
import concurrent.futures
import requests

import openai
import google.generativeai as genai
import anthropic

# ─────────────────────────────────────────────
# KONFIGURASI LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# ENVIRONMENT VARIABLES (API KEYS)
# ─────────────────────────────────────────────
OPENAI_API_KEY    = os.environ["OPENAI_API_KEY"]
GEMINI_API_KEY    = os.environ["GEMINI_API_KEY"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
HUMANIZE_API_KEY  = os.environ["HUMANIZE_API_KEY"]
GITHUB_TOKEN      = os.environ.get("GITHUB_TOKEN", "")   # Opsional untuk rate-limit lebih tinggi

# ─────────────────────────────────────────────
# INISIALISASI SDK
# ─────────────────────────────────────────────
openai_client    = openai.OpenAI(api_key=OPENAI_API_KEY)
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

TODAY = datetime.date.today().isoformat()   # Format: YYYY-MM-DD


# ══════════════════════════════════════════════
# LANGKAH 1 — AMBIL DATA DARI GITHUB REST API
# ══════════════════════════════════════════════
def fetch_trending_python_repos(top_n: int = 5) -> list[dict]:
    """
    Mengambil repo Python yang paling banyak mendapatkan bintang
    dalam 24 jam terakhir via GitHub Search API.
    Mengembalikan list dict berisi informasi ringkas tiap repo.
    """
    log.info("Mengambil data trending repo Python dari GitHub API...")
    url = "https://api.github.com/search/repositories"
    params = {
        "q": f"language:python created:>{TODAY}",
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
        repos = [
            {
                "name":        repo["full_name"],
                "description": repo.get("description") or "Tidak ada deskripsi.",
                "stars":       repo["stargazers_count"],
                "url":         repo["html_url"],
                "topics":      repo.get("topics", []),
                "language":    repo.get("language", "N/A"),
            }
            for repo in items
        ]
        log.info(f"Berhasil mengambil {len(repos)} repo.")
        return repos

    except requests.RequestException as e:
        log.error(f"Gagal mengambil data GitHub: {e}")
        raise


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
# LANGKAH 2 — PANGGIL 3 AI SECARA PARALEL
# ══════════════════════════════════════════════
ANALYSIS_PROMPT_TEMPLATE = """\
Berikut adalah 5 repositori Python yang sedang trending hari ini ({date}):

{repo_text}

Tugasmu:
1. Analisis tren dan tema umum dari kelima repo ini.
2. Identifikasi teknologi atau kebutuhan pasar yang dicerminkan repo-repo tersebut.
3. Berikan ringkasan singkat (maks. 300 kata) yang informatif dan mudah dipahami.
Tuliskan hasilmu dalam Bahasa Indonesia.
"""

def analyze_with_openai(prompt: str) -> str:
    """Memanggil OpenAI GPT-4o-mini untuk analisis."""
    log.info("Memanggil OpenAI GPT-4o-mini...")
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Kamu adalah analis teknologi yang tajam dan komunikatif."},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=600,
            temperature=0.7,
        )
        result = response.choices[0].message.content.strip()
        log.info("OpenAI selesai.")
        return result
    except openai.OpenAIError as e:
        log.error(f"OpenAI error: {e}")
        return f"[OpenAI ERROR] {e}"


def analyze_with_gemini(prompt: str) -> str:
    """Memanggil Google Gemini 1.5 Flash untuk analisis."""
    log.info("Memanggil Google Gemini 1.5 Flash...")
    try:
        response = gemini_model.generate_content(prompt)
        result = response.text.strip()
        log.info("Gemini selesai.")
        return result
    except Exception as e:
        log.error(f"Gemini error: {e}")
        return f"[Gemini ERROR] {e}"


def analyze_with_claude(prompt: str) -> str:
    """Memanggil Anthropic Claude 3 Haiku untuk analisis."""
    log.info("Memanggil Anthropic Claude 3 Haiku...")
    try:
        message = anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=600,
            system="Kamu adalah analis teknologi yang tajam dan komunikatif.",
            messages=[{"role": "user", "content": prompt}],
        )
        result = message.content[0].text.strip()
        log.info("Claude selesai.")
        return result
    except anthropic.APIError as e:
        log.error(f"Claude error: {e}")
        return f"[Claude ERROR] {e}"


def run_parallel_analysis(prompt: str) -> dict[str, str]:
    """
    Menjalankan ketiga fungsi analisis AI secara paralel
    menggunakan ThreadPoolExecutor.
    """
    log.info("Menjalankan 3 analisis AI secara paralel...")
    tasks = {
        "openai": analyze_with_openai,
        "gemini": analyze_with_gemini,
        "claude": analyze_with_claude,
    }
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_map = {executor.submit(fn, prompt): name for name, fn in tasks.items()}
        for future in concurrent.futures.as_completed(future_map):
            name = future_map[future]
            try:
                results[name] = future.result()
            except Exception as e:
                log.error(f"Thread error untuk {name}: {e}")
                results[name] = f"[{name.upper()} THREAD ERROR] {e}"
    log.info("Semua analisis AI selesai.")
    return results


# ══════════════════════════════════════════════
# LANGKAH 3 — OPENAI SEBAGAI JURI
# ══════════════════════════════════════════════
JURY_PROMPT_TEMPLATE = """\
Kamu adalah Juri Senior yang bertugas mengevaluasi tiga analisis berikut tentang
5 repo Python trending hari ini ({date}).

─── ANALISIS DARI OpenAI ───
{openai_result}

─── ANALISIS DARI Google Gemini ───
{gemini_result}

─── ANALISIS DARI Anthropic Claude ───
{claude_result}

Tugasmu:
1. Evaluasi kekuatan dan kelemahan masing-masing analisis.
2. Sintesis satu laporan akhir komprehensif yang menggabungkan poin terbaik dari ketiganya.
3. Gunakan format Markdown yang rapi (heading, bullet point).
4. Panjang laporan akhir: 400–600 kata dalam Bahasa Indonesia.
Laporan akhir harus STANDALONE — pembaca tidak perlu membaca ketiga analisis mentah di atas.
"""

def synthesize_with_jury(ai_results: dict[str, str]) -> str:
    """Meminta OpenAI mensintesis & memilih hasil terbaik dari ketiga AI."""
    log.info("Mengirim hasil ke OpenAI Juri untuk sintesis akhir...")
    jury_prompt = JURY_PROMPT_TEMPLATE.format(
        date=TODAY,
        openai_result=ai_results.get("openai", "-"),
        gemini_result=ai_results.get("gemini", "-"),
        claude_result=ai_results.get("claude", "-"),
    )
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Kamu adalah editor teknologi senior yang menulis laporan berkualitas tinggi."},
                {"role": "user",   "content": jury_prompt},
            ],
            max_tokens=900,
            temperature=0.5,
        )
        result = response.choices[0].message.content.strip()
        log.info("Sintesis akhir dari Juri selesai.")
        return result
    except openai.OpenAIError as e:
        log.error(f"OpenAI Juri error: {e}")
        raise


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
# │  Sesuaikan juga `payload` di dalam fungsi sesuai skema request          │
# │  yang diminta oleh dokumentasi API layanan tersebut.                    │
# └─────────────────────────────────────────────────────────────────────────┘
HUMANIZE_API_ENDPOINT = "https://YOUR-HUMANIZE-AI-ENDPOINT-HERE/v1/humanize"  # ← GANTI INI


def humanize_text(text: str) -> str:
    """
    Mengirim teks ke layanan Humanize AI via HTTP POST.
    Kembalikan teks yang sudah di-humanize, atau teks asli jika gagal.

    Sesuaikan `payload` dengan skema API yang digunakan:
      - Beberapa API menggunakan key "content", "input", atau "text"
      - Beberapa API membutuhkan parameter tambahan seperti "readability", "purpose", dsb.
    """
    log.info("Mengirim teks ke Humanize AI...")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {HUMANIZE_API_KEY}",   # Ubah skema auth jika perlu (misal: "apikey", "x-api-key")
    }
    # ▼ Sesuaikan struktur payload dengan dokumentasi API Anda
    payload = {
        "content": text,          # Beberapa API menggunakan "input" atau "text"
        # "readability": "University",  # Contoh parameter opsional
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

        # ▼ Sesuaikan key response sesuai dokumentasi API Anda
        humanized = (
            data.get("output")          # Coba key "output" terlebih dahulu
            or data.get("result")       # Fallback ke "result"
            or data.get("humanized")    # Fallback ke "humanized"
            or text                     # Jika tidak ditemukan, gunakan teks asli
        )
        log.info("Humanize AI selesai.")
        return humanized

    except requests.RequestException as e:
        log.warning(f"Humanize AI gagal ({e}). Menggunakan teks sintesis asli sebagai fallback.")
        return text   # Graceful fallback: lanjutkan dengan teks asli


# ══════════════════════════════════════════════
# LANGKAH 5 — SIMPAN KE FILE MARKDOWN
# ══════════════════════════════════════════════
def save_to_markdown(content: str, repos: list[dict]) -> str:
    """Menyimpan hasil akhir ke file Markdown dengan nama berisi tanggal."""
    filename = f"hasil_analisis_{TODAY}.md"
    repo_links = "\n".join(
        f"- [{r['name']}]({r['url']}) — ⭐ {r['stars']}" for r in repos
    )
    markdown = f"""# Laporan Analisis Repo Python Trending
**Tanggal:** {TODAY}
**Sumber Data:** GitHub Search API (5 repo Python trending hari ini)

---

## Repo yang Dianalisis

{repo_links}

---

## Hasil Analisis

{content}

---
*Laporan ini dibuat secara otomatis oleh GitHub Actions.*
*Pipeline: GitHub API → OpenAI + Gemini + Claude → OpenAI Juri → Humanize AI*
"""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(markdown)
    log.info(f"Hasil disimpan ke: {filename}")
    return filename


# ══════════════════════════════════════════════
# MAIN ENTRYPOINT
# ══════════════════════════════════════════════
def main():
    log.info("═══ Memulai pipeline analisis AI ═══")

    # 1. Ambil data trending dari GitHub
    repos = fetch_trending_python_repos(top_n=5)
    repo_text = format_repos_for_prompt(repos)

    # 2. Siapkan prompt & jalankan analisis paralel
    analysis_prompt = ANALYSIS_PROMPT_TEMPLATE.format(date=TODAY, repo_text=repo_text)
    ai_results = run_parallel_analysis(analysis_prompt)

    # Debug: log cuplikan tiap hasil AI
    for ai_name, result in ai_results.items():
        log.info(f"[{ai_name}] {result[:80]}...")

    # 3. Sintesis hasil terbaik oleh OpenAI Juri
    final_synthesis = synthesize_with_jury(ai_results)

    # 4. Humanize teks akhir
    humanized_text = humanize_text(final_synthesis)

    # 5. Simpan ke file Markdown
    filename = save_to_markdown(humanized_text, repos)

    log.info(f"═══ Pipeline selesai. File tersimpan: {filename} ═══")


if __name__ == "__main__":
    main()
