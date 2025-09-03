from fastapi.testclient import TestClient
from app.server import app

client = TestClient(app)

def test_health_analyze_no_input():
    r = client.post("/v1/analyze", data={"tasks": "summary"})
    assert r.status_code == 200
    assert r.json()["ok"] is True
