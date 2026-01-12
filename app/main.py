import os
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
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

@app.post("/upload", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):
    # **Label:** Validate file type
    if not file.filename.lower().endswith(".csv"):
        return templates.TemplateResponse("index.html", {
            "request": request,
            "message": "Please upload a valid CSV file."
        })

    try:
        # **Label:** Process attendance
        output_path = process_attendance(file.file)

        # **Label:** Provide download link
        filename = os.path.basename(output_path)
        download_url = f"/download?path={output_path}&name=attendance_final.xlsx"
        return templates.TemplateResponse("index.html", {
            "request": request,
            "message": "File processed successfully.",
            "download_url": download_url
        })
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "message": f"Error: {str(e)}"
        })

@app.get("/download")
async def download(path: str, name: str = "attendance_final.xlsx"):
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, filename=name, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


