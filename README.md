# Bitcoin Hybrid Anomaly Detection

Files:
- data/bitcoin_blockchain_data.csv   (your dataset)
- preprocessing/feature_engineering.py
- models/ (model training code)
- pipeline.py  (train + evaluate + save)
- streamlit_app/app.py  (dashboard)
- saved_models/ (models will be saved here)

Quick start:
1. Activate venv:
   venv\Scripts\activate

2. Install requirements:
   pip install -r requirements.txt

3. Run training + evaluation:
   python pipeline.py

4. Start dashboard:
   streamlit run streamlit_app/app_pipeline.py 

