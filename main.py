from dotenv import load_dotenv

from app.api import create_app

load_dotenv()

from mineru_to_nodes import embedding_model


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
