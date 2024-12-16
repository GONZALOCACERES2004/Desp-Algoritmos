from fastapi import FastAPI, Query
from pydantic import BaseModel
from transformers import pipeline
import random

app = FastAPI()

# Cargar los pipelines de Hugging Face
text_classification_pipeline = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
text_generation_pipeline = pipeline("text-generation", model="gpt2")

# Definir modelos de entrada
class TextRequest(BaseModel):
    text: str

class GenerateRequest(BaseModel):
    prompt: str

@app.get("/")
def read_root():
    """
    Endpoint de bienvenida.
    """
    return {"message": "Bienvenido a pruebas API"}

@app.post("/classify")
def classify_text(request: TextRequest):
    """
    Clasifica el texto proporcionado usando un modelo de Hugging Face.
    """
    result = text_classification_pipeline(request.text)
    return result

@app.post("/generate")
def generate_text(request: GenerateRequest):
    """
    Genera texto basado en el prompt proporcionado usando un modelo de Hugging Face.
    """
    result = text_generation_pipeline(request.prompt, max_length=50)
    return result

@app.get("/random_number")
def get_random_number():
    """
    Genera un número aleatorio entre 1 y 100.
    """
    return {"random_number": random.randint(1, 100)}

@app.get("/multiply")
def multiply(a: float = Query(..., description="First number to multiply"),
             b: float = Query(..., description="Second number to multiply")):
    """
    Multiplica dos números y devuelve el resultado.
    """
    result = a * b
    return {"result": result, "operation": f"{a} * {b} = {result}"}

@app.get("/fibonacci")
def fibonacci(n: int = Query(..., description="Number of Fibonacci sequence elements to generate", ge=1, le=100)):
    """
    Genera una secuencia de Fibonacci con n elementos.
    """
    fib = [0, 1]
    while len(fib) < n:
        fib.append(fib[-1] + fib[-2])
    return {"fibonacci_sequence": fib[:n]}