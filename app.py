from flask import Flask, request, jsonify
from ocr_model import predict
from quran_matcher import match_ayahs
import os

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_route():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    image_path = os.path.join("uploads", image_file.filename)
    os.makedirs("uploads", exist_ok=True)
    image_file.save(image_path)

    try:
        raw_output = predict(image_path)
        matched = match_ayahs(raw_output)

        return jsonify({
            "raw_output": raw_output,
            "matches": [
                {"surah": c, "ayah": v, "text": t} if c != "NO MATCH" else {"no_match": t}
                for c, v, t in matched
            ]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
