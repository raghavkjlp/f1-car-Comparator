import sys
sys.modules["torch.classes"] = None
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import torch
import clip
import math
import pandas as pd
import plotly.graph_objects as go
from streamlit_image_comparison import image_comparison


# ---------------------------
# ---------- SETTINGS -------
# ---------------------------

st.set_page_config(page_title="🏎️ F1 Car Comparator — Ultimate Edition", layout="wide")

st.title("🏎️ F1 Car Comparator ")

st.markdown("""
**✨ Features:**
- ORB + Homography alignment  
- YOLOv8x car segmentation  
- SSIM & CLIP perceptual similarity  
- Part-wise analysis (front / mid / rear)  
- Optical flow visualization  
- AI-style storytelling  
- Leaderboard + CSV Export  
- Interactive image slider  
""")

# ---------------------------
# ----- DEPENDENCY LOAD -----
# ---------------------------
@st.cache_resource
def load_yolo_model(path="yolov8x-seg.pt"):
    return YOLO(path)

@st.cache_resource
def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

yolo = load_yolo_model()
clip_model, clip_preprocess, clip_device = load_clip_model()

# ---------------------------
# ------- UTILITIES ---------
# ---------------------------
def np_img_from_file(f):
    return np.array(Image.open(f).convert("RGB"))

def run_yolo_on_image(img):
    return yolo.predict(img, conf=0.4, imgsz=1280)[0]

def get_largest_mask(result, img_shape):
    h, w = img_shape[:2]
    if result.masks is None or len(result.masks.data) == 0:
        return np.zeros((h, w), dtype=np.uint8)
    masks = result.masks.data.cpu().numpy()
    areas = [np.count_nonzero(m) for m in masks]
    idx = int(np.argmax(areas))
    mask = cv2.resize(masks[idx], (w, h))
    return (mask > 0.5).astype(np.uint8)

def get_largest_mask_box(result, img_shape):
    mask = get_largest_mask(result, img_shape)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return mask, (0, 0, img_shape[1], img_shape[0])
    return mask, (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

def align_orb(img_base, img_to_align, max_feat=3000):
    g1 = cv2.cvtColor(img_base, cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(img_to_align, cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(max_feat)
    k1, d1 = orb.detectAndCompute(g1, None)
    k2, d2 = orb.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return img_to_align, None
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(matcher.match(d1, d2), key=lambda x: x.distance)[:200]
    src = np.float32([k2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    dst = np.float32([k1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        return img_to_align, None
    warped = cv2.warpPerspective(img_to_align, H, (img_base.shape[1], img_base.shape[0]))
    return warped, H

def compute_ssim_map_and_score(A_gray, B_gray):
    try:
        score, diff = ssim(A_gray, B_gray, full=True)
    except Exception:
        A_gray = cv2.resize(A_gray, (256,256))
        B_gray = cv2.resize(B_gray, (256,256))
        score, diff = ssim(A_gray, B_gray, full=True)
    diff_map = (1.0 - diff) * 255
    diff_map = diff_map.astype(np.uint8)
    heat = cv2.applyColorMap(diff_map, cv2.COLORMAP_JET)
    return score, heat

def crop_by_part(img, bbox, part, frac=0.22):
    x1,y1,x2,y2 = bbox
    w = x2 - x1
    if part == "front":
        sx, ex = x1, x1 + int(frac*w)
    elif part == "rear":
        sx, ex = x2 - int(frac*w), x2
    else:
        sx, ex = x1 + int(frac*w), x2 - int(frac*w)
    sx, ex = max(0,sx), min(img.shape[1],ex)
    return img[y1:y2, sx:ex]

def part_ssim(a,b):
    if a.size == 0 or b.size == 0:
        return 0.0, np.zeros((200,200,3), dtype=np.uint8)
    h = 256
    asp = a.shape[1]/max(a.shape[0],1)
    wa = int(h*asp)
    a_r = cv2.resize(a,(wa,h))
    b_r = cv2.resize(b,(wa,h))
    gA = cv2.cvtColor(a_r, cv2.COLOR_RGB2GRAY)
    gB = cv2.cvtColor(b_r, cv2.COLOR_RGB2GRAY)
    sc, heat = compute_ssim_map_and_score(gA,gB)
    return sc, heat

def clip_sim(pilA, pilB):
    device = clip_device
    clip_model.eval()
    with torch.no_grad():
        imA = clip_preprocess(pilA).unsqueeze(0).to(device)
        imB = clip_preprocess(pilB).unsqueeze(0).to(device)
        eA = clip_model.encode_image(imA)
        eB = clip_model.encode_image(imB)
        eA = eA / eA.norm(dim=-1, keepdim=True)
        eB = eB / eB.norm(dim=-1, keepdim=True)
        return float((eA @ eB.T).item())

def optical_flow_arrows(imgA_gray, imgB_gray, mask=None, step=16, scale=2.0):
    flow = cv2.calcOpticalFlowFarneback(imgA_gray, imgB_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    h, w = imgA_gray.shape
    overlay = cv2.cvtColor(imgA_gray, cv2.COLOR_GRAY2BGR)
    for y in range(0, h, step):
        for x in range(0, w, step):
            if mask is not None and mask[y, x] == 0:
                continue
            fx, fy = flow[y, x]
            cv2.arrowedLine(overlay, (x, y), (int(x+scale*fx), int(y+scale*fy)), (0,255,255), 1, tipLength=0.3)
    return overlay

def generate_story(ssim_full, part_ssim_dict, clip_score):
    stories = []
    if ssim_full < 0.9:
        stories.append(f"Overall similarity is {ssim_full*100:.1f}%. Major design evolution detected.")
    for part, score in part_ssim_dict.items():
        if score < 0.9:
            stories.append(f"{part.capitalize()} shows {score*100:.1f}% similarity — possible aerodynamic redesign.")
    if clip_score < 0.9:
        stories.append("CLIP perceptual similarity is lower, suggesting visual identity changes.")
    if not stories:
        stories.append("Cars are nearly identical visually and structurally.")
    return stories

# ---------------------------
# ------- UI / FLOW ---------
# ---------------------------
colA, colB = st.columns(2)
with colA:
    fileA = st.file_uploader("Upload base car image (A)", type=["jpg","jpeg","png"], key="A")
with colB:
    fileB = st.file_uploader("Upload new car image (B)", type=["jpg","jpeg","png"], key="B")

if fileA and fileB:
    imgA = np_img_from_file(fileA)
    imgB = np_img_from_file(fileB)

    st.info("Running segmentation & alignment... please wait ⏳")

    rA = run_yolo_on_image(imgA)
    rB = run_yolo_on_image(imgB)
    maskA, bboxA = get_largest_mask_box(rA, imgA.shape)
    maskB, bboxB = get_largest_mask_box(rB, imgB.shape)

    alignedB, H = align_orb(imgA, imgB)
    if alignedB is None:
        alignedB = imgB.copy()

    # Mask Overlay
    st.subheader("🎯 YOLO Segmentation Overlay")
    if st.checkbox("Show car masks"):
        overlayA = cv2.addWeighted(imgA, 0.7, cv2.cvtColor(maskA*255, cv2.COLOR_GRAY2RGB), 0.3, 0)
        overlayB = cv2.addWeighted(imgB, 0.7, cv2.cvtColor(maskB*255, cv2.COLOR_GRAY2RGB), 0.3, 0)
        st.image([overlayA, overlayB], caption=["Base Car (A)", "New Car (B)"])

    # Before/After slider
    st.subheader("📸 Before / After Comparison")
    image_comparison(img1=imgA, img2=alignedB, label1="Car A", label2="Car B")

    # SSIM
    grayA = cv2.cvtColor(imgA, cv2.COLOR_RGB2GRAY)
    grayB = cv2.cvtColor(alignedB, cv2.COLOR_RGB2GRAY)
    ssim_score, ssim_heat = compute_ssim_map_and_score(grayA, grayB)
    st.metric("Overall SSIM", f"{ssim_score*100:.2f}%")
    st.image(ssim_heat, caption="SSIM Heatmap")

    # Part-wise
    st.subheader("🔎 Part-wise SSIM")
    frontA = crop_by_part(imgA,bboxA,"front")
    midA = crop_by_part(imgA,bboxA,"mid")
    rearA = crop_by_part(imgA,bboxA,"rear")
    frontB = crop_by_part(alignedB,bboxA,"front")
    midB = crop_by_part(alignedB,bboxA,"mid")
    rearB = crop_by_part(alignedB,bboxA,"rear")
    f_sc,f_heat = part_ssim(frontA, frontB)
    m_sc,m_heat = part_ssim(midA, midB)
    r_sc,r_heat = part_ssim(rearA, rearB)
    c1,c2,c3 = st.columns(3)
    c1.metric("Front SSIM",f"{f_sc*100:.2f}%")
    c2.metric("Mid SSIM",f"{m_sc*100:.2f}%")
    c3.metric("Rear SSIM",f"{r_sc*100:.2f}%")
    st.image([frontA, frontB, f_heat], width=220)
    st.image([midA, midB, m_heat], width=220)
    st.image([rearA, rearB, r_heat], width=220)

    # CLIP similarity
    st.subheader("🤖 CLIP Similarity")
    clip_score = clip_sim(Image.fromarray(imgA), Image.fromarray(alignedB))
    st.metric("Perceptual Similarity", f"{clip_score*100:.2f}%")

    # Radar Chart
    st.subheader("📊 Performance Radar")
    metrics = ['Front','Mid','Rear','CLIP']
    values = [f_sc, m_sc, r_sc, clip_score]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values, theta=metrics, fill='toself', name='Similarity'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Optical Flow
    st.subheader("💨 Optical Flow Visualization")
    flow_overlay = optical_flow_arrows(grayA, grayB, maskA)
    st.image(flow_overlay, caption="Motion of design features")

    # Storytelling
    st.subheader("📝 AI-style Insights")
    part_ssim_dict = {"front": f_sc, "mid": m_sc, "rear": r_sc}
    for line in generate_story(ssim_score, part_ssim_dict, clip_score):
        st.write("• " + line)

    # History + Benchmark
    if "history" not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append({
        "Car Pair": f"{fileA.name} vs {fileB.name}",
        "SSIM": round(ssim_score,3),
        "CLIP": round(clip_score,3)
    })
    st.subheader("🏁 Benchmark History")
    df_hist = pd.DataFrame(st.session_state.history)
    st.dataframe(df_hist)

    # Export CSV
    csv = df_hist.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Results CSV", csv, "car_comparison.csv", "text/csv")

else:
    st.info("Upload both car images to begin comparison.")
