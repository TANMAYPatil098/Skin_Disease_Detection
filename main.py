from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import os
from predict import predict_skin_disease
from fastapi.responses import JSONResponse

app = FastAPI()

# Define allowed origins
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1:5500",
    "http://127.0.0.1:3000",
    "http://your-frontend-domain.com"
]

# Apply CORS middleware with correct settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Changed from [""] to ["*"]
    allow_headers=["*"],  # Changed from [""] to ["*"]
)

# Create upload directory
UPLOAD_DIR = "temp_images/"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            return JSONResponse(
                content={"error": "File must be an image"},
                status_code=400
            )

        # Save the uploaded image temporarily
        image_path = os.path.join(UPLOAD_DIR, file.filename)
        print(f"Saving uploaded image to: {image_path}")

        # Read the file content
        file_content = await file.read()
        
        # Save the file
        with open(image_path, "wb") as buffer:
            buffer.write(file_content)

        # Verify file was saved
        if not os.path.exists(image_path):
            return JSONResponse(
                content={"error": "Failed to save image"},
                status_code=500
            )

        try:
            # Get prediction
            prediction = predict_skin_disease(image_path)
            print(f"Prediction for {file.filename}: {prediction}")
        except Exception as pred_error:
            return JSONResponse(
                content={"error": f"Prediction error: {str(pred_error)}"},
                status_code=500
            )
        finally:
            # Clean up: remove temporary file
            if os.path.exists(image_path):
                os.remove(image_path)
                print(f"Deleted image: {image_path}")

        return JSONResponse(content={"prediction": prediction})

    except Exception as e:
        print(f"Error during upload: {str(e)}")
        return JSONResponse(
            content={"error": f"Upload failed: {str(e)}"},
            status_code=500
        )

# Add a test endpoint
@app.get("/")
async def root():
    return {"message": "API is running"}
