# Exporta datos a JSON para Streamlit Cloud
import json

def export_to_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
