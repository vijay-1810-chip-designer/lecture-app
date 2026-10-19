import os, re, shutil
from pathlib import Path
from xml.sax.saxutils import escape

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from anthropic import Anthropic
from supabase import create_client
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

DATA = Path("data")
DATA.mkdir(exist_ok=True)
BUCKET = "lecture-notes"

openai_client = OpenAI()          # reads OPENAI_API_KEY
claude = Anthropic()              # reads ANTHROPIC_API_KEY
supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

app = FastAPI()


def safe_name(name: str) -> str:
    name = re.sub(r"[^\w\- ]+", "", name).strip().replace(" ", "_")
    return name or "lecture"


@app.post("/api/chunk")
async def chunk(session_id: str = Form(...), index: int = Form(...),
                language: str = Form(""), audio: UploadFile = File(...)):
    """Receive one ~5 min audio segment, transcribe it, save the text."""
    sid = safe_name(session_id)
    folder = DATA / sid
    folder.mkdir(exist_ok=True)
    raw = await audio.read()
    kwargs = {"model": "whisper-1", "file": (audio.filename, raw, audio.content_type)}
    if language:
        kwargs["language"] = language  # e.g. "en", "hi", "ta"
    result = openai_client.audio.transcriptions.create(**kwargs)
    (folder / f"{index:05d}.txt").write_text(result.text, encoding="utf-8")
    return {"ok": True, "index": index, "chars": len(result.text)}


def build_pdf(path: Path, title: str, transcript: str, summary: str | None):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(path), pagesize=A4, title=title)
    story = [Paragraph(escape(title), styles["Title"]), Spacer(1, 12)]
    if summary:
        story += [Paragraph("Summary", styles["Heading2"]),
                  Paragraph(escape(summary).replace("\n", "<br/>"), styles["BodyText"]),
                  Spacer(1, 18)]
    story.append(Paragraph("Full Transcript", styles["Heading2"]))
    for para in transcript.split("\n\n"):
        story += [Paragraph(escape(para), styles["BodyText"]), Spacer(1, 6)]
    doc.build(story)


@app.post("/api/finish")
async def finish(session_id: str = Form(...), title: str = Form(...),
                 summarize: bool = Form(True)):
    sid = safe_name(session_id)
    folder = DATA / sid
    parts = sorted(folder.glob("*.txt"))
    if not parts:
        raise HTTPException(400, "No transcript found for this session")
    transcript = "\n\n".join(p.read_text(encoding="utf-8").strip() for p in parts)

    summary = None
    if summarize:
        msg = claude.messages.create(
            model="claude-sonnet-5-5",
            max_tokens=2000,
            messages=[{"role": "user", "content":
                "Summarize this lecture transcript as clear study notes: key topics, "
                "definitions, and important points. Use short bullet points.\n\n" + transcript}],
        )
        summary = msg.content[0].text

    filename = safe_name(title) + ".pdf"
    pdf_path = folder / filename
    build_pdf(pdf_path, title, transcript, summary)

    supabase.storage.from_(BUCKET).upload(
        filename, pdf_path.read_bytes(),
        {"content-type": "application/pdf", "upsert": "true"})
    shutil.rmtree(folder, ignore_errors=True)
    return {"filename": filename, "url": signed_url(filename)}


def signed_url(filename: str) -> str:
    res = supabase.storage.from_(BUCKET).create_signed_url(filename, 3600)
    return res["signedURL"]


@app.get("/api/files")
def files():
    items = supabase.storage.from_(BUCKET).list()
    return [{"name": i["name"], "url": signed_url(i["name"])} for i in items
            if i["name"].endswith(".pdf")]


@app.post("/api/rename")
def rename(old_name: str = Form(...), new_title: str = Form(...)):
    new_name = safe_name(new_title) + ".pdf"
    supabase.storage.from_(BUCKET).move(old_name, new_name)
    return {"filename": new_name, "url": signed_url(new_name)}


app.mount("/", StaticFiles(directory="../frontend", html=True), name="static")
