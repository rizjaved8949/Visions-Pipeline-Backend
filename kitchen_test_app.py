from fastapi import FastAPI

from kitchen_monitoring.api import (
    router as kitchen_router,
)


app = FastAPI(
    title="Kitchen PPE Test API",
)


app.include_router(
    kitchen_router,
    prefix="/api/kitchen",
    tags=[
        "Kitchen Hygiene"
    ],
)


if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "kitchen_test_app:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
    )