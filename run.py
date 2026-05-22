import uvicorn

from app.config import DEBUG, HOST, PORT


def main():
    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=DEBUG)


if __name__ == "__main__":
    main()
