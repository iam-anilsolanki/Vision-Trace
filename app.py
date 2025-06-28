import os
import cv2
import torch
import numpy as np
import streamlit as st
from datetime import timedelta
from sklearn.metrics.pairwise import cosine_similarity
import torchvision.transforms as T
import torchreid
from PIL import Image
from ultralytics import YOLO
import torch.nn.modules.container
import ultralytics.nn.modules

st.set_page_config(page_title="VisionTrace", layout="wide")
st.title("🔍 VisionTrace: Person Finder in CCTV Footage")

# Upload query image
query_image_file = st.file_uploader("Upload image of the person to search for", type=["jpg", "jpeg", "png"])
video_files = st.file_uploader("Upload CCTV videos", type=["mp4"], accept_multiple_files=True)

SIMILARITY_THRESHOLD = st.slider("Similarity threshold", 0.3, 1.0, 0.75, 0.01)

if query_image_file and video_files:
    # Prepare output
    os.makedirs("results", exist_ok=True)

    # Monkey-patch torch.load to allow full checkpoint deserialization (for YOLOv8 weights)
    original_load = torch.load
    def patched_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return original_load(*args, **kwargs)
    torch.load = patched_load

    # Load models
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    yolo = YOLO('yolov8n.pt')
    torch.load = original_load

    def load_model(model_name='osnet_x1_0'):
        model = torchreid.models.build_model(
            name=model_name,
            num_classes=1000,
            loss='softmax',
            pretrained=True
        )
        model.eval()
        return model.to(device)

    def preprocess(img, height=256, width=128):
        transform = T.Compose([
            T.ToPILImage(),
            T.Resize((height, width)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        return transform(img).unsqueeze(0)

    def extract_features(img):
        with torch.no_grad():
            input_tensor = preprocess(img).to(device)
            features = reid_model(input_tensor)
            return features.cpu().numpy()

    # Load ReID model
    reid_model = load_model()

    # Read query image
    query_bytes = np.frombuffer(query_image_file.read(), np.uint8)
    query_img = cv2.imdecode(query_bytes, cv2.IMREAD_COLOR)
    query_feat = extract_features(query_img)

    found_matches = []

    for video_file in video_files:
        st.markdown(f"### 📹 Processing {video_file.name}")
        tfile = open(f"temp_{video_file.name}", 'wb')
        tfile.write(video_file.read())
        tfile.close()

        cap = cv2.VideoCapture(f"temp_{video_file.name}")
        frame_idx = 0
        fps = cap.get(cv2.CAP_PROP_FPS)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % 10 != 0:
                continue

            results = yolo(frame)[0]
            for box in results.boxes:
                cls = int(box.cls[0])
                if cls != 0:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                person_crop = frame[y1:y2, x1:x2]

                try:
                    feat = extract_features(person_crop)
                    sim = cosine_similarity(query_feat, feat)[0][0]

                    if sim > SIMILARITY_THRESHOLD:
                        timestamp = str(timedelta(seconds=frame_idx / fps))
                        st.success(f"🎯 Match found in {video_file.name} at {timestamp} | Score: {sim:.2f}")
                        st.image(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB), caption=f"{video_file.name} @ {timestamp}", width=300)
                        found_matches.append((video_file.name, timestamp, sim))
                except Exception as e:
                    st.warning(f"⚠️ Error processing frame {frame_idx}: {e}")
                    continue

        cap.release()
        os.remove(f"temp_{video_file.name}")

    if not found_matches:
        st.info("No matches found.")
    else:
        st.success(f"✅ Completed! Total matches: {len(found_matches)}")
else:
    st.info("Upload both query image and videos to begin.")
