from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field

from model import DEFAULT_WEIGHTS_PATH, IMAGE_SIZE, NZ, load_generator

app = FastAPI(
    title='Flowers102 DCGAN Microservice',
    description='Generates a synthetic 64x64 flower image from a latent noise vector using the DCGAN generator from the uploaded notebook.',
    version='1.0.0',
)

MODEL_PATH = Path(os.getenv('MODEL_PATH', DEFAULT_WEIGHTS_PATH))
OUTPUT_DIR = Path(os.getenv('OUTPUT_DIR', 'generated_samples'))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
model, model_status = load_generator(MODEL_PATH)


class GenerateRequest(BaseModel):
    seed: Optional[int] = Field(default=42, description='Random seed used to create the latent vector.')
    save_file: bool = Field(default=False, description='Whether to save the generated image on the server.')


class GenerateResponse(BaseModel):
    seed: int
    image_size: int
    model_status: str
    image_base64: str
    file_path: Optional[str] = None


@app.get('/')
def root() -> dict:
    return {
        'message': 'Flowers102 DCGAN microservice is running.',
        'docs_url': '/docs',
        'predict_endpoint': '/generate',
        'model_status': model_status,
    }


@app.get('/health')
def health() -> dict:
    return {'status': 'ok', 'model_status': model_status}


@app.post('/generate', response_model=GenerateResponse)
def generate(req: GenerateRequest) -> JSONResponse:
    if req.seed is None:
        raise HTTPException(status_code=400, detail='seed must be an integer or omitted.')

    g = torch.Generator(device='cpu')
    g.manual_seed(req.seed)
    noise = torch.randn((1, NZ, 1, 1), generator=g)

    with torch.no_grad():
        output = model(noise).detach().cpu().squeeze(0)

    # Convert from [-1,1] to [0,255]
    output = ((output + 1.0) / 2.0).clamp(0, 1)
    output = (output.permute(1, 2, 0).numpy() * 255).astype('uint8')
    image = Image.fromarray(output)

    buf = io.BytesIO()
    image.save(buf, format='PNG')
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    file_path = None
    if req.save_file:
        file_path = OUTPUT_DIR / f'flower_seed_{req.seed}.png'
        image.save(file_path)
        file_path = file_path.as_posix()

    payload = GenerateResponse(
        seed=req.seed,
        image_size=IMAGE_SIZE,
        model_status=model_status,
        image_base64=image_base64,
        file_path=file_path,
    )
    return JSONResponse(content=payload.model_dump())
