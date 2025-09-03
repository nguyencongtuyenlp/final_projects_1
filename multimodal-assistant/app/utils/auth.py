from typing import Optional
import os
from fastapi import Request, HTTPException, status

AUTH_ENV = "APP_AUTH_TOKEN"

def require_bearer(request: Request):
    token = os.getenv(AUTH_ENV, "").strip()
    if not token:
        return  # auth disabled
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")
    if auth.split(" ", 1)[1].strip() != token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
