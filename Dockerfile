# Hugging Face Spaces (Docker) image for the Delivery Station Planner app.
FROM python:3.11-slim

# Spaces run the container as user 1000; give it a home and a writable working folder.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements-app.txt .
RUN pip install --no-cache-dir --user -r requirements-app.txt

# The app, the engine, the assumptions and the small prepared data and saved results it loads.
COPY --chown=user assumptions.yaml .
COPY --chown=user .streamlit ./.streamlit
COPY --chown=user app ./app
COPY --chown=user engine ./engine
COPY --chown=user data/prepared ./data/prepared
COPY --chown=user scenarios ./scenarios

EXPOSE 7860
# Spaces show the app inside an iframe on huggingface.co, so cross-site checks are switched off.
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0", \
     "--server.headless=true", "--server.enableXsrfProtection=false", "--server.enableCORS=false", \
     "--browser.gatherUsageStats=false"]
