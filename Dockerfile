FROM python:3.9

WORKDIR /app

COPY requirements.txt ./
RUN pip install -U pip
RUN pip install --no-cache-dir -r requirements.txt

COPY chat_engine.py /usr/local/lib/python3.9/site-packages/canopy/chat_engine

COPY . ./

CMD [ "streamlit", "run", "./main.py" ]