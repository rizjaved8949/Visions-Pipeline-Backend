# run_api.py  — backend-only launcher, no frontend
import os
from dotenv import load_dotenv

load_dotenv()

from Attendance import server

from guard_monitoring.api import router as guard_monitoring_router
server.app.include_router(guard_monitoring_router, prefix="/api/guard", tags=["Guard Monitoring"])

from kitchen_monitoring.api import router as kitchen_router
server.app.include_router(kitchen_router, prefix="/api/kitchen", tags=["Kitchen Hygiene"])

from restricted_zone_monitor.api import router as restricted_zone_router
server.app.include_router(restricted_zone_router, prefix="/api/restricted-zone", tags=["Restricted Zone Monitor"])

from sample_videos import router as sample_videos_router
server.app.include_router(sample_videos_router, prefix="/api/sample-videos", tags=["Sample Videos"])

app = server.app  # uvicorn needs this name at module level