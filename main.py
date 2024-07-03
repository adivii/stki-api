from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from utils import clustering
from utils import semantics

app = FastAPI()
clustering.load_model()
clustering.load_data()

origins = [
    "http://localhost",
    "http://127.0.0.1:8000",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


@app.get("/cari_jurnal")
async def cluster(keyword: str):
    return semantics.search_similar_titles(keyword)
