import io
from typing import cast

import torch
from diffusers import AutoPipelineForText2Image  # type: ignore
from fastapi import FastAPI, Response, UploadFile
from transformers import pipeline

app = FastAPI()

audio_transcriptor = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-base",
    kwargs={"language": "french", "task": "transcribe"},
    device="mps",
)

prompt_generator = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-1.5B-Instruct",
    return_full_text=False,
    device="mps",
)

image_generator = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sd-turbo",
    torch_dtype=torch.bfloat16,
    variant="fp16",
).to("mps")


async def transcribe_audio(file: UploadFile):

    audio_input = await file.read()
    result = cast(dict[str, str], audio_transcriptor(audio_input))

    return result["text"]


def generate_prompt(transcription: str):
    messages = [
        {
            "role": "system",
            "content": "Tu es un robot traducteur strict spécialisé dans Stable Diffusion. Ta seule fonction est de traduire la description française de l'utilisateur en un prompt en anglais, ajuste le prompt pour que l'image soit réalise avec du détails. Règle absolue : Ne renvoie QUE la traduction en anglais. N'ajoute aucune formule de politesse, aucune introduction, aucune conclusion.",
        },
        {"role": "user", "content": transcription},
    ]
    generated = prompt_generator(messages, max_new_tokens=70, do_sample=False)

    return cast(str, generated[0]["generated_text"])


def generate_image(prompt: str):

    image = image_generator(
        prompt=prompt, num_inference_steps=4, guidance_scale=0.0
    ).images[0]  # type: ignore

    return image


@app.post("/generate_image")
async def generate_image_endpoint(description: UploadFile):
    transcribed_audio = await transcribe_audio(description)
    print(f"Transcribed audio: {transcribed_audio}")

    generated_prompt = generate_prompt(transcribed_audio)
    print(f"Generated promt: {generated_prompt}")

    generated_image = generate_image(generated_prompt)

    memory_stream = io.BytesIO()
    generated_image.save(memory_stream, format="PNG")
    image_bytes = memory_stream.getvalue()

    return Response(content=image_bytes, media_type="image/png")
