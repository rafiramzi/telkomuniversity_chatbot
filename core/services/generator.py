import cohere
import os

# v2 client untuk chat_stream (yang sudah lo pakai sebelumnya)
co_v2 = cohere.ClientV2(os.getenv("COHERE_API_KEY"))

# v1 client untuk grounded RAG (co.chat dengan documents=...).
# v1 punya native document grounding yang lebih akurat untuk RAG ketat.
co_v1 = cohere.Client(os.getenv("COHERE_API_KEY"))


# ---------- Preambles ----------

_LATEX_RULES = (
    "FORMAT MATEMATIKA — ATURAN KETAT:\n"
    "Setiap ekspresi matematika WAJIB dibungkus dengan delimiter berikut.\n"
    "Inline (di tengah kalimat): \\( ... \\)\n"
    "Block (rumus berdiri sendiri di baris terpisah): \\[ ... \\]\n\n"
    "JANGAN PERNAH gunakan:\n"
    "- Tanda dolar $ ... $ atau $$ ... $$  (TIDAK didukung renderer)\n"
    "- Backtick atau code block untuk rumus\n"
    "- Plain text untuk simbol matematika\n\n"
    "Contoh BENAR:\n"
    "  IPK dihitung dengan rumus \\( \\text{IPK} = \\frac{\\sum (\\text{Nilai} \\times \\text{SKS})}{\\sum \\text{SKS}} \\).\n"
    "  Block: \\[ \\text{IKK} = \\frac{N_i}{N_m} \\times 4 \\]\n\n"
    "Contoh SALAH (JANGAN DITIRU):\n"
    "  $\\text{IPK} = ...$           ← pakai $ dilarang\n"
    "  \\text{IPK} = \\frac{...}{...} ← tidak ada delimiter pembungkus\n"
    "  IPK = (Σ Nilai × SKS) / Σ SKS ← plain text, bukan LaTeX\n\n"
    "Aturan tambahan:\n"
    "- Pakai \\frac{a}{b} untuk pembagian (JANGAN tulis a/b)\n"
    "- Pakai \\times untuk perkalian (JANGAN tulis x atau *)\n"
    "- Pakai \\sqrt{x} untuk akar (JANGAN tulis sqrt(x))\n"
    "- Pakai \\sum untuk sigma (JANGAN tulis Σ atau 'jumlah')\n"
    "- Subscript pakai underscore dengan kurung kurawal: N_{m}, bukan Nm\n"
    "- Tulis \\text{NamaVariabel} kalau nama variabelnya kata, mis. \\text{SKS}\n"
)


_STRICT_PREAMBLE = (
    "Anda adalah asisten akademik Telkom University yang membantu mahasiswa "
    "memahami peraturan, kebijakan, dan prosedur akademik. "
    "Jawab dalam Bahasa Indonesia dengan informatif dan membantu.\n\n"

    "ATURAN UTAMA — TOPIK:\n"
    "1. Sapaan, basa-basi, dan percakapan ringan (mis. 'halo', 'hai', 'apa kabar', "
    "   'terima kasih', 'siapa kamu') → balas RAMAH dan SINGKAT sebagai asisten "
    "   akademik TelU. Tidak perlu merujuk dokumen.\n"
    "2. Pertanyaan seputar Telkom University → jawab berdasarkan DOKUMEN yang diberikan.\n"
    "   - Dokumen berisi jawaban langsung: berikan jawaban lengkap dan rinci.\n"
    "   - Dokumen berisi info terkait tapi tidak persis: jelaskan yang ada, "
    "     sebutkan bahwa detail spesifik tidak ditemukan, tawarkan topik lain.\n"
    "   - Dokumen tidak relevan sama sekali: katakan informasi tidak tersedia "
    "     dalam data yang dimiliki.\n"
    "3. Pertanyaan di LUAR Telkom University (politik, hiburan, resep, berita "
    "   umum, sains umum, dll.) → TOLAK sopan: \"Maaf, saya hanya dapat membantu "
    "   pertanyaan seputar Telkom University.\"\n\n"

    "LARANGAN KERAS:\n"
    "- JANGAN menggunakan pengetahuan umum atau internet untuk menjawab "
    "  pertanyaan substantif. Hanya boleh dari dokumen yang diberikan.\n"
    "- JANGAN mengarang fakta, angka, atau kebijakan yang tidak ada di dokumen.\n"
    "- JANGAN mencampur singkatan berbeda makna: "
    "  IPK (Indeks Prestasi Kumulatif), IPS (Indeks Prestasi Semester), "
    "  SKS (Satuan Kredit Semester), TAK (Transkrip Aktivitas Kemahasiswaan), "
    "  IKK (Indeks Keaktifan Kumulatif — terkait TAK, BUKAN IPK).\n\n"

    "FORMAT JAWABAN:\n"
    "- Gunakan bullet points untuk daftar, tabel Markdown untuk data tabular.\n"
    "- JANGAN menambahkan daftar 'Sumber' di akhir jawaban.\n\n"
    + _LATEX_RULES
)


_LOOSE_PREAMBLE = (
    "Anda adalah asisten akademik Telkom University. "
    "Jawab dalam Bahasa Indonesia.\n\n"

    "ATURAN UTAMA — TOPIK:\n"
    "1. Sapaan dan basa-basi → balas RAMAH dan SINGKAT.\n"
    "2. Pertanyaan seputar Telkom University → jawab berdasarkan konteks dokumen "
    "   yang diberikan. Jangan mengarang fakta di luar dokumen.\n"
    "3. Pertanyaan di luar Telkom University → TOLAK sopan: \"Maaf, saya hanya "
    "   dapat membantu pertanyaan seputar Telkom University.\"\n\n"
    + _LATEX_RULES
)


# ---------- LaTeX delimiter sanitizer ----------
# Walaupun prompt sudah jelas, model kadang masih mengeluarkan rumus
# dengan delimiter $...$ atau bahkan tanpa pembungkus sama sekali.
# Sanitizer ini bekerja sebagai jaring pengaman setelah streaming.
#
# Catatan penting: karena outputnya streaming token-by-token, kita TIDAK
# bisa mengubah delimiter di tengah stream (token mungkin terpotong).
# Strategi: buffer satu baris pada satu waktu, sanitasi baris itu, lalu
# yield-kan baris yang sudah bersih.

import re as _re


def _sanitize_latex_line(line: str) -> str:
    """
    Bersihkan satu baris teks:
    1. Konversi $$ ... $$ → \\[ ... \\] (block math)
    2. Konversi $ ... $   → \\( ... \\) (inline math)
    3. Bungkus baris yang BERISI command LaTeX tapi tidak ada delimiter
       pembungkus utuh dengan \\[ ... \\] (best-effort).
    """
    # 1) $$ ... $$ → \[ ... \]
    line = _re.sub(r"\$\$(.+?)\$\$", r"\\[\1\\]", line)

    # 2) $ ... $ → \( ... \)  (hati-hati: jangan match $$ yang sudah diproses)
    line = _re.sub(r"(?<!\$)\$([^\$\n]+?)\$(?!\$)", r"\\(\1\\)", line)

    # 3) Cari LaTeX command yang muncul DI LUAR pasangan delimiter manapun.
    #    Cara sederhana: hapus dulu semua segmen \( ... \) dan \[ ... \],
    #    lalu cek apakah masih ada command tersisa di luar.
    stripped_segments = _re.sub(r"\\\([^\n]*?\\\)", "", line)
    stripped_segments = _re.sub(r"\\\[[^\n]*?\\\]", "", stripped_segments)

    has_orphan_cmd = bool(_re.search(
        r"\\(?:frac|sum|sqrt|times|cdot|int|prod|partial|"
        r"alpha|beta|gamma|delta|theta|sigma|mu|pi|lambda|"
        r"leq|geq|neq|approx|infty)\b",
        stripped_segments
    ))
    # \text{...} sendirian sering muncul di prose normal (mis. dalam diskusi),
    # jadi kita hanya trigger auto-wrap kalau ada operator math (frac/sum/dll)
    # ATAU \text{} yang berdampingan dengan = atau operator.
    if not has_orphan_cmd:
        has_orphan_cmd = bool(_re.search(
            r"\\text\{[^}]*\}\s*[=+\-]",
            stripped_segments
        ))

    if has_orphan_cmd:
        stripped = line.strip()
        leading_ws = line[:len(line) - len(line.lstrip())]
        line = f"{leading_ws}\\[ {stripped} \\]"

    return line


def _sanitize_stream(token_iter):
    """
    Wrapper generator: terima iterator token streaming, kumpulkan per baris,
    sanitasi, lalu yield. Token tetap stream-friendly: non-newline part
    di-yield apa adanya, hanya saat ketemu '\\n' baris di-flush setelah
    disanitasi.
    """
    buffer = ""
    for tok in token_iter:
        if tok is None:
            continue
        buffer += tok
        # Flush setiap baris lengkap
        while "\n" in buffer:
            line, _, rest = buffer.partition("\n")
            yield _sanitize_latex_line(line) + "\n"
            buffer = rest
    # Flush sisa terakhir
    if buffer:
        yield _sanitize_latex_line(buffer)


# ---------- MODEL 2: Grounded RAG dengan streaming ----------

import re as _re2

# Pola sapaan / chitchat yang tidak butuh dokumen.
# Normalisasi huruf berulang dulu sebelum match (haloo → halo, hiii → hi).
_CHITCHAT_PATTERNS = [
    r"^h[ai]+\b",           # hi, hai, hai, hiii, dll
    r"^hal+o+\b",           # halo, haloo, halooo
    r"^hel+o+\b",           # hello, helloo
    r"^hey+\b",             # hey, heyy
    r"^hei+\b",             # hei, heii
    r"^selamat\b",          # selamat pagi/siang/sore/malam
    r"^(pagi|siang|sore|malam)\b",
    r"apa kabar",
    r"^(makasih|terima kasih|thanks|thank you|thx)\b",
    r"siapa (kamu|anda|lo|elu)",
    r"(kamu|lo|elu) siapa",
    r"^(oke|ok|sip|mantap|wkwk|hehe|lol)\b",
]

_CHITCHAT_SYSTEM = (
    "Anda adalah asisten akademik Telkom University yang ramah. "
    "Balas sapaan dan basa-basi dengan singkat dan hangat dalam Bahasa Indonesia. "
    "Perkenalkan diri sebagai asisten akademik TelU bila ditanya. "
    "Jangan membahas topik di luar Telkom University."
)


def _is_chitchat(query: str) -> bool:
    """True kalau query adalah sapaan atau basa-basi yang tidak butuh dokumen."""
    q = query.lower().strip().rstrip("!.,?~")
    # Collapse huruf berulang: haloo→halo, hiii→hi
    q = _re2.sub(r"(.)\1{2,}", r"\1\1", q)
    q = _re2.sub(r"([aeiou])\1+\b", r"\1", q)
    # Query panjang hampir pasti bukan chitchat
    if len(q.split()) > 8:
        return False
    for pat in _CHITCHAT_PATTERNS:
        if _re2.search(pat, q):
            return True
    return False


def generate_grounded_answer_stream(query, chunks, weak_match=False, is_enumeration=False):
    """
    Streaming answer dengan Cohere grounded RAG (model2).

    `chunks` = list of dict, masing-masing minimal punya:
        { "id": str, "text": str, "category": str (opsional), "source": str (opsional) }

    `weak_match` = True kalau retrieval merasa match-nya lemah
    (relevance score rerank rendah). Saat True, kita inject hint ke
    message supaya LLM eksplisit menyebut bahwa informasi spesifik
    tidak ditemukan dan menawarkan topik terkait yang ADA di dokumen.

    `is_enumeration` = True kalau query minta daftar/enumerasi
    (mis. "apa saja", "semua", "lokasi-lokasi"). Saat True, kita
    perintahkan LLM untuk MENGUMPULKAN informasi dari SEMUA dokumen
    yang diberikan, bukan fokus pada satu saja.

    Mengembalikan generator of plain text.
    """
    # ------------------------------------------------------------------
    # Bypass grounded RAG untuk sapaan / chitchat.
    # Cohere grounded mode sangat literal: kalau dokumen tidak mengandung
    # jawaban untuk "hi", model mengabaikan preamble dan bilang "tidak ada".
    # Untuk chitchat, cukup pakai co_v2 tanpa documents=.
    # ------------------------------------------------------------------
    if _is_chitchat(query):
        try:
            stream = co_v2.chat_stream(
                model="command-a-03-2025",
                messages=[
                    {"role": "system", "content": _CHITCHAT_SYSTEM},
                    {"role": "user", "content": query},
                ],
                temperature=0.7,
            )

            def _raw_chitchat():
                for event in stream:
                    if event.type == "content-delta":
                        try:
                            text = event.delta.message.content.text
                            if text:
                                yield text
                        except Exception:
                            pass

            for clean_chunk in _sanitize_stream(_raw_chitchat()):
                yield clean_chunk
        except Exception as e:
            yield f"\n[STREAM ERROR] {str(e)}"
        return

    if not chunks:
        yield "Maaf, informasi tersebut tidak tersedia dalam data yang saya miliki."
        return

    # Bentuk dokumen sesuai format yang dimengerti Cohere v1 chat RAG
    documents = []
    for c in chunks:
        documents.append({
            "id": str(c.get("id", "")),
            "title": c.get("source", c.get("category", "Dokumen")),
            "snippet": c["text"],
        })

    # Susun hint untuk LLM berdasarkan flag
    hints = []

    if is_enumeration:
        hints.append(
            "Pertanyaan ini meminta DAFTAR atau ENUMERASI. "
            "Tolong: (1) periksa SEMUA dokumen yang diberikan, "
            "(2) kumpulkan SEMUA item yang relevan dari setiap dokumen, "
            "(3) susun jawaban dalam bentuk daftar lengkap. "
            "Jangan menjawab hanya berdasarkan satu dokumen kalau "
            "dokumen lain juga berisi item yang dimaksud."
        )

    if weak_match:
        hints.append(
            "Pencarian tidak menemukan jawaban langsung untuk pertanyaan di atas. "
            "Tolong: (1) sampaikan secara eksplisit bahwa informasi spesifik "
            "yang ditanyakan tidak ditemukan di dokumen, "
            "(2) jelaskan topik-topik TERKAIT yang ADA di dokumen yang diberikan, "
            "(3) sarankan pertanyaan lain yang bisa dijawab."
        )

    if hints:
        effective_message = f"{query}\n\n[Catatan untuk asisten: " + " ".join(hints) + "]"
    else:
        effective_message = query

    try:
        stream = co_v1.chat_stream(
            model="command-r-plus-08-2024",
            message=effective_message,
            documents=documents,
            preamble=_STRICT_PREAMBLE,
            temperature=0.4,
        )

        def _raw():
            for event in stream:
                # Cohere v1 streaming events: 'stream-start', 'text-generation',
                # 'citation-generation', 'stream-end'.
                etype = getattr(event, "event_type", None)
                if etype == "text-generation":
                    text = getattr(event, "text", None)
                    if text:
                        yield text

        for clean_chunk in _sanitize_stream(_raw()):
            yield clean_chunk
    except Exception as e:
        yield f"\n[STREAM ERROR] {str(e)}"


# ---------- MODEL 1: Loose/contextual chat (dipertahankan apa adanya) ----------

def generate_answer_stream(query, context, strict=False):
    """
    Versi lama yang dipakai model1 (dan fallback untuk model2 kalau dibutuhkan).
    Tetap pakai ClientV2 chat_stream dengan plain prompt stuffing.
    """

    if strict:
        system_prompt = (
            _STRICT_PREAMBLE
            + "\n\nKonteks dokumen:\n"
            + context
        )
    else:
        system_prompt = (
            _LOOSE_PREAMBLE
            + "\n\nKonteks dokumen:\n"
            + context
        )

    try:
        stream = co_v2.chat_stream(
            model="command-a-03-2025",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            temperature=0.3 if strict else 0.7,
        )

        def _raw():
            for event in stream:
                if event.type == "content-delta":
                    try:
                        text = event.delta.message.content.text
                        if text:
                            yield text
                    except Exception:
                        pass

        for clean_chunk in _sanitize_stream(_raw()):
            yield clean_chunk

    except Exception as e:
        yield f"\n[STREAM ERROR] {str(e)}"