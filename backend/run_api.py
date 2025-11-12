"""Convenience script to run the ASGI app with uvicorn (API + Django)."""
import os
import uvicorn


if __name__ == '__main__':
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ai_tripod_backend.settings')
    host = os.getenv('API_HOST', '0.0.0.0')
    port = int(os.getenv('API_PORT', '8000'))
    print(f"Starting API on {host}:{port}")
    uvicorn.run('ai_tripod_backend.asgi:application', host=host, port=port, reload=True)
