import os
import torch
import numpy as np
import random
from flask import Flask, render_template, request
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageEnhance
from flask_sqlalchemy import SQLAlchemy
import pickle
from datetime import datetime
import logging

# -------------------- SEED --------------------
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# -------------------- LOGGING --------------------
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# -------------------- DATABASE --------------------
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///faces.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# -------------------- MODEL --------------------
class Face(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    embedding = db.Column(db.LargeBinary, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# -------------------- MODEL SETUP --------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=False, device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# -------------------- GLOBALS --------------------
known_faces = {}
SIMILARITY_THRESHOLD = 0.55

# -------------------- PREPROCESS --------------------
def preprocess_image(img):
    img = ImageEnhance.Contrast(img).enhance(1.3)
    img = ImageEnhance.Brightness(img).enhance(1.1)
    return img

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# -------------------- LOAD DB --------------------
def load_faces_from_db():
    global known_faces
    known_faces = {}
    faces = Face.query.all()

    for face in faces:
        emb = pickle.loads(face.embedding)
        known_faces.setdefault(face.name, []).append(emb)

def load_dataset_to_db():
    dataset_path = os.getenv("DATASET_PATH", "dataset/train")

    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)

        if not os.path.isdir(person_folder):
            continue

        logging.info(f"Loading {person_name}...")

        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)

            try:
                img = preprocess_image(Image.open(img_path).convert('RGB'))
                face_tensor = mtcnn(img)

                if face_tensor is None:
                    continue

                with torch.no_grad():
                    emb = facenet(face_tensor.unsqueeze(0).to(device)).squeeze().cpu().numpy()

                emb = emb / np.linalg.norm(emb)

                db.session.add(Face(
                    name=person_name,
                    embedding=pickle.dumps(emb)
                ))

            except Exception as e:
                logging.error(f"Error processing {img_path}: {e}")

    db.session.commit()
    logging.info("Dataset loaded")

# -------------------- IDENTIFICATION --------------------
def identify_person(file):
    if not known_faces:
        return "Database Empty"

    img = preprocess_image(Image.open(file).convert('RGB'))
    face_tensor = mtcnn(img)

    if face_tensor is None:
        return "No Face Detected"

    with torch.no_grad():
        emb = facenet(face_tensor.unsqueeze(0).to(device)).squeeze().cpu().numpy()

    emb = emb / np.linalg.norm(emb)

    best_name, best_sim, second_best = None, -1, -1

    for name, emb_list in known_faces.items():
        sims = [cosine_similarity(emb, e) for e in emb_list]
        sim = max(sims)

        if sim > best_sim:
            second_best = best_sim
            best_sim = sim
            best_name = name
        elif sim > second_best:
            second_best = sim

    margin = best_sim - second_best

    if best_sim >= SIMILARITY_THRESHOLD:
        if margin < 0.04:
            return f"Ambiguous | Sim:{best_sim:.3f} | Margin:{margin:.3f}"
        return f"{best_name} | Sim:{best_sim:.3f} | Margin:{margin:.3f}"

    return "Unknown"

# -------------------- ROUTE --------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file1 = request.files.get('image1')
        file2 = request.files.get('image2')

        result1 = identify_person(file1) if file1 else "Upload Image"
        result2 = identify_person(file2) if file2 else "Upload Image"

        return render_template('upload_index.html', result1=result1, result2=result2)

    return render_template('upload_index.html', result1=None, result2=None)

# -------------------- INIT --------------------
with app.app_context():
    db.create_all()

    if Face.query.count() == 0:
        load_dataset_to_db()

    load_faces_from_db()

if __name__ == '__main__':
    app.run(debug=True)