FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir pandas numpy scikit-learn joblib matplotlib pydantic ollama streamlit xgboost
COPY src ./src
CMD ["streamlit", "run", "src/app.py", "--server.address=0.0.0.0"]
