# Policy Doctor — combined-demo Streamlit app.
#
# Build:  docker build -t policy-doctor-demo .
# Run:    docker run -p 8501:8501 policy-doctor-demo
# Open:   http://localhost:8501
#
# The image bundles:
#   - the full policy_doctor package
#   - all pre-computed clusterings for transport_mh_jan28 (47+ variants)
#   - the rendered rollout MP4s used by the user-study pages
#   - the user-study session YAMLs

FROM python:3.10-slim

# System deps for plotting + video frame extraction.
# - libgl1: cv2 / imageio image decoding (needed by mp4 thumbnail extraction)
# - libglib2.0-0: Qt-style image plugins
# - ffmpeg: imageio video reading
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better Docker layer caching).
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the policy_doctor package + third_party clusterings.
COPY policy_doctor /app/policy_doctor
COPY third_party /app/third_party

# MP4s for the user-study + graph-demo pages.
COPY data/study_mp4s /tmp/study_mp4s

# Demo sweep clustering results.
COPY data/demo_sweep /app/data/demo_sweep

ENV PYTHONPATH=/app
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "/app/policy_doctor/streamlit_app/demo_app/Home.py", \
     "--server.port", "8501", \
     "--server.address", "0.0.0.0", \
     "--server.headless", "true"]
