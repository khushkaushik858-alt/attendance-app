import os
import io
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from app.processing import process_attendance

app = FastAPI(title="Attendance Processing")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "message": None})

@app.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...)):
    # **Label:** Validate file type
    if not file.filename.lower().endswith(".csv"):
        return templates.TemplateResponse("index.html", {
            "request": request,
            "message": "Please upload a valid CSV file."
        })

    try:
        # **Label:** Process attendance (returns Excel bytes)
        content = process_attendance(file.file)

        # **Label:** Stream file back to client
        return StreamingResponse(
            io.BytesIO(content),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": "attachment; filename=attendance_final.xlsx"
            }
        )
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "message": f"Error: {str(e)}"
        })