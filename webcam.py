import cv2
import torch
import numpy as np
from flask import Flask, render_template, Response, jsonify
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

# IMPORT DB + CONTEXT
from img_upload import load_faces_from_db, known_faces, app as upload_app

app = Flask(__name__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=True, device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

SIMILARITY_THRESHOLD = 0.55

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        face_tensor = mtcnn(img)

        label = "No Face"

        if face_tensor is not None and known_faces:
            if face_tensor.ndim == 4:
                face_tensor = face_tensor[0]

            with torch.no_grad():
                emb = facenet(face_tensor.unsqueeze(0).to(device)).squeeze().cpu().numpy()

            emb = emb / np.linalg.norm(emb)

            best_name, best_sim = None, -1

            for name, emb_list in known_faces.items():
                sims = [cosine_similarity(emb, e) for e in emb_list]
                sim = max(sims)

                if sim > best_sim:
                    best_sim = sim
                    best_name = name

            if best_sim >= SIMILARITY_THRESHOLD:
                label = f"{best_name} ({best_sim:.2f})"
            else:
                label = "Unknown"

        cv2.putText(frame, label, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# 🔥 LOAD DB ONLY WHEN ROUTE IS HIT (SAFE)
@app.route('/')
def index():
    with upload_app.app_context():
        load_faces_from_db()

    return render_template('webcam_index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_name')
def get_name():
    return jsonify({"status": "Running"})

if __name__ == '__main__':
    app.run(debug=True)