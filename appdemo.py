import streamlit as st
import numpy as np
import joblib
import pickle
import tempfile, os, io, time, warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle
from scipy.ndimage import uniform_filter

st.set_page_config(
    page_title="ThermaLens — Delhi UHI Predictor",
    page_icon="assets/icon.png" if os.path.exists("assets/icon.png") else None,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ═══════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

*, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
}

/* ── page background ── */
.main, .block-container {
    background: #F2F4F7 !important;
    padding-top: 0 !important;
}

/* ── hero ── */
.hero-wrap {
    background: #0B1D3A;
    border-radius: 20px;
    padding: 52px 56px 48px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero-wrap::before {
    content: '';
    position: absolute; top: -120px; right: -80px;
    width: 420px; height: 420px; border-radius: 50%;
    background: radial-gradient(circle, rgba(56,189,248,0.12) 0%, transparent 70%);
}
.hero-wrap::after {
    content: '';
    position: absolute; bottom: -80px; left: 30%;
    width: 280px; height: 280px; border-radius: 50%;
    background: radial-gradient(circle, rgba(16,185,129,0.08) 0%, transparent 70%);
}
.app-label {
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.14em;
    color: #38BDF8; text-transform: uppercase; margin-bottom: 12px;
}
.hero-title {
    font-size: 2.8rem; font-weight: 700; color: #F8FAFC;
    letter-spacing: -0.5px; line-height: 1.15; margin: 0 0 14px;
}
.hero-title span { color: #38BDF8; }
.hero-sub {
    font-size: 1.05rem; color: #94A3B8; line-height: 1.7;
    max-width: 600px; margin: 0 0 28px;
}
.hero-pills { display: flex; gap: 10px; flex-wrap: wrap; }
.pill {
    display: inline-block; padding: 5px 16px; border-radius: 20px;
    font-size: 0.75rem; font-weight: 600; letter-spacing: 0.03em;
    border: 1px solid rgba(255,255,255,0.1);
}
.pill-blue  { background: rgba(56,189,248,0.12); color: #7DD3FC; }
.pill-green { background: rgba(16,185,129,0.12); color: #6EE7B7; }
.pill-amber { background: rgba(251,191,36,0.12); color: #FDE68A; }

/* ── cards ── */
.card {
    background: white; border-radius: 16px; padding: 24px 26px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.04);
    margin-bottom: 14px;
}
.card-label {
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: #94A3B8; margin-bottom: 14px;
}

/* ── step pills ── */
.step-row {
    display: flex; align-items: flex-start; gap: 12px;
    padding: 10px 0; border-bottom: 1px solid #F1F5F9;
}
.step-row:last-child { border-bottom: none; }
.step-num {
    width: 26px; height: 26px; border-radius: 8px;
    background: #0B1D3A; color: white;
    font-size: 0.75rem; font-weight: 700;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0; margin-top: 1px;
}
.step-text { font-size: 0.84rem; color: #475569; line-height: 1.5; }
.step-text b { color: #1E293B; }

/* ── upload hint ── */
.upload-hint {
    background: #F0F9FF; border: 1.5px dashed #7DD3FC;
    border-radius: 10px; padding: 14px 16px;
    font-size: 0.78rem; color: #0369A1; margin-top: 10px; line-height: 1.9;
    font-family: 'DM Mono', monospace !important;
}

/* ── metric strip ── */
.mstrip { display: flex; gap: 10px; margin: 20px 0; }
.mcard {
    flex: 1; background: white; border-radius: 14px; padding: 18px 12px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    border-top: 3px solid #E2E8F0;
}
.mcard.c0 { border-top-color: #3B82F6; }
.mcard.c1 { border-top-color: #F59E0B; }
.mcard.c2 { border-top-color: #EF4444; }
.mcard.cs { border-top-color: #64748B; }
.mval { font-size: 1.7rem; font-weight: 700; line-height: 1; }
.mlbl { font-size: 0.63rem; color: #94A3B8; margin-top: 5px;
        text-transform: uppercase; letter-spacing: 0.07em; }
.v0  { color: #3B82F6; }
.v1  { color: #D97706; }
.v2  { color: #EF4444; }
.vs  { color: #64748B; }

/* ── class legend cards ── */
.cls-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin-top: 16px; }
.cls-card {
    border-radius: 14px; padding: 20px 18px;
    border: 1.5px solid;
}
.cls-card.cool {
    background: linear-gradient(160deg, #EFF6FF, #DBEAFE);
    border-color: #BFDBFE;
}
.cls-card.mod {
    background: linear-gradient(160deg, #FFFBEB, #FEF3C7);
    border-color: #FDE68A;
}
.cls-card.hot {
    background: linear-gradient(160deg, #FFF5F5, #FEE2E2);
    border-color: #FECACA;
}
.cls-dot {
    width: 10px; height: 10px; border-radius: 50%;
    display: inline-block; margin-right: 6px;
}
.cls-title {
    font-size: 1rem; font-weight: 700; margin-bottom: 6px;
    display: flex; align-items: center;
}
.cls-temp { font-size: 0.75rem; font-weight: 600;
            padding: 2px 10px; border-radius: 12px;
            display: inline-block; margin-bottom: 8px; }
.cls-desc { font-size: 0.82rem; line-height: 1.6; }
.cool .cls-title { color: #1E40AF; }
.cool .cls-dot   { background: #3B82F6; }
.cool .cls-temp  { background: #DBEAFE; color: #1D4ED8; }
.cool .cls-desc  { color: #1E3A5F; }
.mod  .cls-title { color: #92400E; }
.mod  .cls-dot   { background: #F59E0B; }
.mod  .cls-temp  { background: #FEF3C7; color: #B45309; }
.mod  .cls-desc  { color: #78350F; }
.hot  .cls-title { color: #991B1B; }
.hot  .cls-dot   { background: #EF4444; }
.hot  .cls-temp  { background: #FEE2E2; color: #B91C1C; }
.hot  .cls-desc  { color: #7F1D1D; }

/* ── result box ── */
.rbox {
    border-radius: 18px; padding: 36px 32px; text-align: center;
    margin-bottom: 18px; border: 2px solid;
}
.r0 { background: linear-gradient(160deg,#DBEAFE,#EFF6FF); border-color:#BFDBFE; }
.r1 { background: linear-gradient(160deg,#FEF3C7,#FFFBEB); border-color:#FDE68A; }
.r2 { background: linear-gradient(160deg,#FEE2E2,#FFF5F5); border-color:#FECACA; }
.r-indicator {
    width: 14px; height: 14px; border-radius: 50%;
    display: inline-block; margin-right: 10px;
}
.r-class  { font-size: 2rem; font-weight: 800; margin: 12px 0 8px; }
.r-desc   { font-size: 0.92rem; opacity: 0.8; line-height: 1.6; }
.rc0 { color: #1E40AF; } .rc1 { color: #92400E; } .rc2 { color: #991B1B; }

/* ── progress bars ── */
.pbar-wrap { background: #F1F5F9; border-radius: 6px; height: 7px;
             margin: 5px 0; overflow: hidden; }
.pbar-fill { height: 100%; border-radius: 6px; transition: width 0.4s; }

/* ── empty state ── */
.empty-state {
    background: white; border-radius: 18px; padding: 72px 40px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.empty-icon  {
    width: 72px; height: 72px; border-radius: 20px;
    background: #F0F9FF; margin: 0 auto 20px;
    display: flex; align-items: center; justify-content: center;
    font-size: 2rem;
}
.empty-title { font-size: 1.2rem; font-weight: 700; color: #1E293B; margin-bottom: 8px; }
.empty-sub   { color: #94A3B8; font-size: 0.88rem; line-height: 1.7; max-width: 340px; margin: 0 auto; }

/* ── GEE guide ── */
.gee-step {
    background: white; border-radius: 12px; padding: 18px 20px;
    margin-bottom: 10px; border-left: 4px solid #38BDF8;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.gee-step h4 { margin: 0 0 5px; color: #0B1D3A; font-size: 0.92rem; }
.gee-step p  { margin: 0; font-size: 0.82rem; color: #475569; line-height: 1.6; }

/* ── divider ── */
.divider {
    height: 1px; background: linear-gradient(90deg, transparent, #E2E8F0, transparent);
    margin: 24px 0;
}

/* ── table ── */
table { width: 100%; border-collapse: collapse; }
th { background: #F8FAFC; color: #64748B; font-size: 0.75rem;
     font-weight: 600; letter-spacing: 0.05em; padding: 8px 12px;
     text-transform: uppercase; }
td { font-size: 0.82rem; color: #334155; padding: 8px 12px;
     border-bottom: 1px solid #F1F5F9; }

/* ── tab styling ── */
div[data-testid="stTabs"] button {
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    color: #64748B !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #0B1D3A !important;
}

/* ── footer ── */
.footer {
    text-align: center; padding: 32px 0 12px;
    font-size: 0.72rem; color: #CBD5E1;
    letter-spacing: 0.06em; text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════
WINDOWS = [3, 7, 11, 15, 21, 31]
CLASSES = ["Cool", "Moderate", "Hot"]
HEX     = ["#3B82F6", "#F59E0B", "#EF4444"]
R_CSS   = ["r0","r1","r2"]
R_CLS   = ["rc0","rc1","rc2"]
R_IND   = ["#3B82F6","#F59E0B","#EF4444"]
DESC    = [
    "This area has lower surface temperatures. It is typically covered by trees, grass, water bodies, or open fields that naturally absorb less heat.",
    "This area has average surface temperatures — typical of residential neighbourhoods, mixed-use zones, and suburban areas with a blend of buildings and green patches.",
    "This area has significantly elevated surface temperatures. Dense buildings, roads, concrete, and industrial activity trap heat and raise temperatures well above the city average.",
]
DELHI   = {"lon_min":76.84,"lon_max":77.35,"lat_min":28.40,"lat_max":28.88}
MAX_DIM = 600

# ═══════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_model():
    import requests, pickle
    
    # Force re-download scaler with correct version
    for f in ["/tmp/scaler_download.pkl"]:
        if os.path.exists(f):
            os.remove(f)
            
    HF_BASE = "https://huggingface.co/Drishtanta23/thermallens_delhi_uhi/resolve/main"
    files = {
        "Delhi_UHI_XGB_Stage5.pkl": f"{HF_BASE}/Delhi_UHI_XGB_Stage5.pkl",
        "scaler_download.pkl":      f"{HF_BASE}/scaler_download.pkl",
    }

    for fname, url in files.items():
        local = f"/tmp/{fname}"
        if not os.path.exists(local):
            print(f"Downloading {fname}...")
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(local, "wb") as f:
                    for chunk in r.iter_content(8192):
                        f.write(chunk)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = joblib.load("/tmp/Delhi_UHI_XGB_Stage5.pkl")
            with open("/tmp/scaler_download.pkl", "rb") as f:
                scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        st.error(f"Model load failed: {e}")
        st.stop()
# ═══════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════
def sf(a, w):
    return uniform_filter(np.where(np.isfinite(a), a, 0.0), w)

def sstd(a, w):
    f = np.where(np.isfinite(a), a, 0.0)
    return np.sqrt(np.clip(uniform_filter(f**2,w)-uniform_filter(f,w)**2, 0, None))

def scale_sr(arr):
    a  = arr.astype(float)
    fv = a[np.isfinite(a)]
    if len(fv) == 0: return a
    med = np.median(fv)
    if med > 1000:
        a = np.where((np.isfinite(a))&(a>1000), a*0.0000275-0.2, np.nan)
    elif med > 1:
        a = np.where(np.isfinite(a), a/10000.0, np.nan)
    return np.clip(np.where(np.isfinite(a), a, np.nan), 0, 1)

def build_features(b2,b3,b4,b5,b6,b7):
    eps    = 1e-6
    NDVI   = (b5-b4)/(b5+b4+eps)
    NDBI   = (b6-b5)/(b6+b5+eps)
    NDBSI  = (b6+b4-b5-b2)/(b6+b4+b5+b2+eps)
    Albedo = 0.356*b2+0.130*b4+0.373*b5+0.085*b6+0.072*b7-0.018
    BUFrac = np.clip((NDBI+1)/2, 0, 1)
    DEM    = np.where(np.isfinite(b2), 0.0, np.nan)
    raw = [NDVI,NDBI,NDBSI,BUFrac,Albedo,DEM,
           NDBI-NDVI, NDVI*(1-BUFrac), BUFrac*(1-Albedo),
           (NDVI>0.3).astype(float)*(1-BUFrac),
           (1-Albedo)*BUFrac*NDBI,
           np.abs(NDBI-sf(NDBI,7))/(np.abs(NDBI)+eps)]
    spat = [sf(a,w) for a in [NDVI,NDBI,BUFrac,Albedo] for w in WINDOWS]
    tex  = [sstd(a,w) for a in [NDVI,BUFrac] for w in [7,15,21]]
    diff = ([sf(a,wa)-sf(a,wb) for a in [NDBI,BUFrac]
             for wa,wb in [(3,7),(7,11),(11,21),(21,31)]] +
            [sf(NDVI,wa)-sf(NDVI,wb) for wa,wb in [(3,7),(7,21),(21,31)]])
    H,W = b2.shape
    X   = np.stack([f.ravel() for f in raw+spat+tex+diff], axis=1)
    valid = (
        np.isfinite(NDVI.ravel())&np.isfinite(NDBI.ravel())&
        np.isfinite(Albedo.ravel())&
        (NDVI.ravel()>=-1)&(NDVI.ravel()<=1)&
        (NDBI.ravel()>=-1)&(NDBI.ravel()<=1)
    )
    X[valid] = np.nan_to_num(X[valid], nan=0.0, posinf=0.0, neginf=0.0)
    return X, valid, H, W, NDVI, NDBI, Albedo

def run_pred(X, valid, H, W, model, scaler):
    labels = np.full(H*W, -1, dtype=np.int8)
    if valid.sum() > 0:
        Xs = X.copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Xs[valid] = scaler.transform(X[valid])
        labels[valid] = model.predict(Xs[valid]).astype(np.int8)
    return labels.reshape(H, W)

# ═══════════════════════════════════════════════════════════
# GEOTIFF
# ═══════════════════════════════════════════════════════════
def read_tiff(path):
    try:
        import rasterio
        from rasterio.warp import transform_bounds
        from rasterio.crs import CRS
        from rasterio.coords import BoundingBox
        with rasterio.open(path) as src:
            if src.count < 7:
                return None, f"Need at least 7 bands, found {src.count}.", None, None
            data = src.read().astype(np.float64)
            if src.nodata is not None:
                data[data == src.nodata] = np.nan
            b = src.bounds; crs = src.crs
            if crs and crs.to_epsg() != 4326:
                l,bot,r,t = transform_bounds(
                    crs, CRS.from_epsg(4326),
                    b.left, b.bottom, b.right, b.top)
                b = BoundingBox(l, bot, r, t)
        return data, None, None, b
    except Exception as e:
        return None, str(e), None, None

def downsample(data, max_dim=MAX_DIM):
    from scipy.ndimage import zoom as sz
    H,W = data.shape[1], data.shape[2]
    if max(H,W) <= max_dim: return data, 1.0
    f = max_dim/max(H,W)
    nH,nW = int(H*f), int(W*f)
    out = np.zeros((data.shape[0],nH,nW), dtype=data.dtype)
    for b in range(data.shape[0]):
        out[b] = sz(data[b], (nH/H, nW/W), order=1, prefilter=False)
    return out, f

def latlon_to_px(lat, lon, bounds, H, W):
    if bounds is None: return None, None
    col = int((lon-bounds.left)/(bounds.right-bounds.left)*W)
    row = int((bounds.top-lat)/(bounds.top-bounds.bottom)*H)
    return (row,col) if 0<=row<H and 0<=col<W else (None,None)

# ═══════════════════════════════════════════════════════════
# MAPS
# ═══════════════════════════════════════════════════════════
def draw_map(label_map, bounds, ndvi):
    fig, axes = plt.subplots(1, 2, figsize=(16,6.5), dpi=130)
    fig.patch.set_facecolor("#F2F4F7")

    cmap = ListedColormap(["#0D1B2A","#3B82F6","#F59E0B","#EF4444"])
    disp = np.where(label_map==-1, 0, label_map+1)
    ext  = ([bounds.left,bounds.right,bounds.bottom,bounds.top]
            if bounds else None)

    # LEFT — UHI classification
    ax = axes[0]; ax.set_facecolor("#0D1B2A")
    kw = dict(cmap=cmap,vmin=0,vmax=3,interpolation="nearest",aspect="auto")
    if ext:
        ax.imshow(disp, extent=ext, **kw)
        ax.set_xlabel("Longitude", fontsize=9, color="#64748B")
        ax.set_ylabel("Latitude",  fontsize=9, color="#64748B")
        ax.add_patch(Rectangle(
            (DELHI["lon_min"],DELHI["lat_min"]),
            DELHI["lon_max"]-DELHI["lon_min"],
            DELHI["lat_max"]-DELHI["lat_min"],
            lw=1.5, edgecolor="white", facecolor="none", ls="--", alpha=0.5))
        for nm,(lo,la) in {
                "Connaught Place":(77.216,28.632),
                "Ridge Forest":(77.130,28.700),
                "Okhla":(77.270,28.550),
                "Yamuna":(77.280,28.660)}.items():
            if bounds.left<lo<bounds.right and bounds.bottom<la<bounds.top:
                ax.annotate(nm, xy=(lo,la), fontsize=6.5, color="white",
                            ha="center",
                            bbox=dict(boxstyle="round,pad=0.25",
                                      fc="#0B1D3A",alpha=0.8,ec="none"))
    else:
        ax.imshow(disp, **kw)

    ax.set_title("UHI Classification", fontsize=11, fontweight="bold",
                 color="#0B1D3A", pad=10, loc="left")
    ax.tick_params(colors="#94A3B8", labelsize=8)
    for s in ax.spines.values(): s.set_edgecolor("#E2E8F0")
    ax.legend(
        handles=[mpatches.Patch(color=c,label=l)
                 for c,l in zip(HEX, CLASSES)],
        loc="lower left", fontsize=8, framealpha=0.92,
        facecolor="white", edgecolor="#E2E8F0")

    # RIGHT — NDVI
    ax2 = axes[1]; ax2.set_facecolor("#0D1B2A")
    nd  = np.where(label_map==-1, np.nan, ndvi)
    kw2 = dict(cmap="RdYlGn", vmin=-0.2, vmax=0.6,
               interpolation="bilinear", aspect="auto")
    im  = ax2.imshow(nd, **({"extent":ext,**kw2} if ext else kw2))
    if ext:
        ax2.set_xlabel("Longitude", fontsize=9, color="#64748B")
        ax2.set_ylabel("Latitude",  fontsize=9, color="#64748B")
    ax2.set_title("Vegetation Index (NDVI)", fontsize=11,
                  fontweight="bold", color="#0B1D3A", pad=10, loc="left")
    ax2.tick_params(colors="#94A3B8", labelsize=8)
    for s in ax2.spines.values(): s.set_edgecolor("#E2E8F0")
    cb = plt.colorbar(im, ax=ax2, shrink=0.82, pad=0.02)
    cb.set_label("NDVI  (higher = more vegetation)", fontsize=8, color="#64748B")
    cb.ax.tick_params(labelsize=8, colors="#64748B")

    fig.suptitle("Delhi NCT  —  Urban Heat Island Analysis  |  Landsat 9, Summer 2023",
                 fontsize=11, fontweight="600", color="#64748B", y=1.01)
    plt.tight_layout()
    return fig

# ═══════════════════════════════════════════════════════════
# HERO
# ═══════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-wrap">
  <div class="app-label">ThermaLens — Urban Heat Intelligence</div>
  <div class="hero-title">
    How hot is<br><span>your city?</span>
  </div>
  <div class="hero-sub">
    Upload a Landsat 9 satellite image of Delhi to instantly map
    every neighbourhood as Cool, Moderate, or Hot — using a machine
    learning model trained on 192,000 pixels with 94% accuracy.
  </div>
  <div class="hero-pills">
    <span class="pill pill-blue">XGBoost Stage 5</span>
    <span class="pill pill-green">94.01% Accuracy</span>
    <span class="pill pill-amber">Landsat 9 · Delhi 2023</span>
    <span class="pill pill-blue">53 Spatial Features</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── what are the classes? ──
st.markdown("""
<div class="cls-grid">
  <div class="cls-card cool">
    <div class="cls-title">
      <span class="cls-dot"></span> Cool Zone
    </div>
    <div class="cls-temp">Below city average temperature</div>
    <div class="cls-desc">
      Areas with trees, parks, water bodies, or farmland. Natural surfaces
      absorb less heat and release it slowly, keeping temperatures lower.
      Examples: Yamuna floodplain, Ridge Forest, Lodhi Garden.
    </div>
  </div>
  <div class="cls-card mod">
    <div class="cls-title">
      <span class="cls-dot"></span> Moderate Zone
    </div>
    <div class="cls-temp">Around city average temperature</div>
    <div class="cls-desc">
      Mixed residential and suburban areas with a blend of buildings,
      roads, and some green patches. Temperatures are neither extreme
      nor low. Examples: most residential colonies across Delhi NCT.
    </div>
  </div>
  <div class="cls-card hot">
    <div class="cls-title">
      <span class="cls-dot"></span> Hot Zone
    </div>
    <div class="cls-temp">Significantly above city average</div>
    <div class="cls-desc">
      Dense urban cores, industrial areas, and heavily paved zones. Concrete
      and asphalt absorb heat all day and radiate it at night, creating
      dangerous hotspots. Examples: Central Delhi, Okhla, Shahdara.
    </div>
  </div>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "  Predict — Upload Satellite Image  ",
    "  Point Lookup — Enter Coordinates  ",
    "  Data Guide — Get from GEE  ",
])

# ═══════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ═══════════════════════════════════════════════════════════
with tab1:
    lc, rc = st.columns([1, 2.6], gap="large")

    with lc:
        st.markdown("""
        <div class="card">
          <div class="card-label">How to use</div>
          <div class="step-row">
            <div class="step-num">1</div>
            <div class="step-text">
              Export your Landsat 9 image from Google Earth Engine
              (see the <b>Data Guide tab</b> for the exact script)
            </div>
          </div>
          <div class="step-row">
            <div class="step-num">2</div>
            <div class="step-text">
              Upload the downloaded <b>.tif file</b> using the button below
            </div>
          </div>
          <div class="step-row">
            <div class="step-num">3</div>
            <div class="step-text">
              Wait ~10–15 seconds for the model to classify every pixel
            </div>
          </div>
          <div class="step-row">
            <div class="step-num">4</div>
            <div class="step-text">
              View the UHI map and vegetation index, then download as PNG
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Upload Landsat 9 GeoTIFF",
            type=["tif","tiff"],
            label_visibility="visible")

        st.markdown("""
        <div class="upload-hint">
          Expected band order:<br>
          Band 1  Blue (SR_B2)<br>
          Band 2  Green (SR_B3)<br>
          Band 3  Red (SR_B4)<br>
          Band 4  NIR (SR_B5)<br>
          Band 5  SWIR1 (SR_B6)<br>
          Band 6  SWIR2 (SR_B7)<br>
          Band 7  Thermal (ST_B10)
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card" style="margin-top:14px;">
          <div class="card-label">Model Details</div>
          <table>
            <tr><td>Algorithm</td><td><b>XGBoost Stage 5</b></td></tr>
            <tr><td>Accuracy</td><td><b>94.01%</b></td></tr>
            <tr><td>Features</td><td><b>53 multi-scale spatial</b></td></tr>
            <tr><td>Training data</td><td><b>Delhi NCT, Summer 2023</b></td></tr>
            <tr><td>Processing time</td><td><b>10–15 seconds</b></td></tr>
          </table>
        </div>
        """, unsafe_allow_html=True)

    with rc:
        if uploaded is None:
            st.markdown("""
            <div class="empty-state">
              <div class="empty-icon">&#x1F6F0;</div>
              <div class="empty-title">No image uploaded yet</div>
              <div class="empty-sub">
                Upload a Landsat 9 GeoTIFF to generate a pixel-level
                heat map of Delhi. The model will classify every valid
                pixel as Cool, Moderate, or Hot in under 15 seconds.
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            t0   = time.time()
            prog = st.progress(0, text="Reading satellite image...")

            with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
                tmp.write(uploaded.read()); tmp_path = tmp.name
            data, err, _, bounds = read_tiff(tmp_path)
            os.unlink(tmp_path)
            if err: st.error(f"Error: {err}"); st.stop()

            H0,W0 = data.shape[1], data.shape[2]
            prog.progress(10, text=f"Downsampling {W0}x{H0} for speed...")
            data, factor = downsample(data, MAX_DIM)
            H,W = data.shape[1], data.shape[2]

            prog.progress(22, text="Scaling spectral bands...")
            b2=scale_sr(data[0]); b3=scale_sr(data[1]); b4=scale_sr(data[2])
            b5=scale_sr(data[3]); b6=scale_sr(data[4]); b7=scale_sr(data[5])

            prog.progress(38, text="Computing 53 spatial features...")
            X,valid,H,W,NDVI,NDBI,Albedo = build_features(b2,b3,b4,b5,b6,b7)

            if valid.sum() == 0:
                fv = b2[np.isfinite(b2)]
                st.error("No valid pixels found. Check band order and "
                         "that this is a Landsat Collection 2 Level-2 product.")
                if len(fv):
                    st.caption(f"Band 2 range: {fv.min():.1f} to {fv.max():.1f}, "
                               f"median {np.median(fv):.1f}")
                st.stop()

            prog.progress(60, text="Running XGBoost model...")
            model, scaler = load_model()
            label_map = run_pred(X, valid, H, W, model, scaler)

            st.session_state.update({
                "label_map":label_map, "bounds":bounds,
                "H":H, "W":W, "NDVI":NDVI
            })

            elapsed = time.time() - t0
            prog.progress(88, text="Rendering maps...")

            total = int(valid.sum())
            nc = int((label_map[label_map>=0]==0).sum())
            nm = int((label_map[label_map>=0]==1).sum())
            nh = int((label_map[label_map>=0]==2).sum())

            st.markdown(f"""
            <div class="mstrip">
              <div class="mcard cs">
                <div class="mval vs">{total:,}</div>
                <div class="mlbl">Pixels classified</div>
              </div>
              <div class="mcard c0">
                <div class="mval v0">{nc/total*100:.1f}%</div>
                <div class="mlbl">Cool zones</div>
              </div>
              <div class="mcard c1">
                <div class="mval v1">{nm/total*100:.1f}%</div>
                <div class="mlbl">Moderate zones</div>
              </div>
              <div class="mcard c2">
                <div class="mval v2">{nh/total*100:.1f}%</div>
                <div class="mlbl">Hot zones</div>
              </div>
              <div class="mcard cs">
                <div class="mval vs">{elapsed:.0f}s</div>
                <div class="mlbl">Processing time</div>
              </div>
            </div>""", unsafe_allow_html=True)

            if factor < 1.0:
                st.info(f"Image downsampled from {W0}x{H0} to {W}x{H} pixels "
                        f"for faster processing. Spatial patterns are preserved.")

            prog.progress(95)
            fig = draw_map(label_map, bounds, NDVI.reshape(H,W))
            st.pyplot(fig); plt.close(fig)
            prog.progress(100, text=f"Done in {elapsed:.1f} seconds.")

            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
            st.markdown("""
            <div class="card">
              <div class="card-label">Reading the map</div>
              <p style="font-size:0.85rem;color:#475569;line-height:1.7;margin:0;">
                <b style="color:#3B82F6;">Blue areas</b> are Cool zones — parks, forests, and water bodies
                that stay naturally cooler. <b style="color:#D97706;">Amber areas</b> are Moderate zones —
                everyday residential and mixed neighbourhoods. <b style="color:#EF4444;">Red areas</b> are
                Hot zones — the urban heat islands, concentrated around dense commercial and industrial
                areas. The black regions are outside Delhi's boundary or were masked due to cloud cover.
              </p>
            </div>""", unsafe_allow_html=True)

            buf = io.BytesIO()
            fig2, ax2 = plt.subplots(figsize=(10,8), dpi=150)
            cmap2 = ListedColormap(["#0D1B2A","#3B82F6","#F59E0B","#EF4444"])
            ax2.imshow(np.where(label_map==-1,0,label_map+1),
                       cmap=cmap2, vmin=0, vmax=3,
                       interpolation="nearest", aspect="auto",
                       **({"extent":[bounds.left,bounds.right,
                                     bounds.bottom,bounds.top]}
                          if bounds else {}))
            ax2.set_title("UHI Classification — Delhi NCT 2023", fontweight="bold")
            ax2.legend(handles=[mpatches.Patch(color=c,label=l)
                                  for c,l in zip(HEX,CLASSES)],
                       loc="lower left", fontsize=9)
            plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
            plt.close(fig2); buf.seek(0)
            st.download_button(
                "Download UHI Map (PNG)",
                data=buf,
                file_name="thermallens_delhi_uhi.png",
                mime="image/png")

# ═══════════════════════════════════════════════════════════
# TAB 2 — POINT LOOKUP
# ═══════════════════════════════════════════════════════════
with tab2:
    lc2, rc2 = st.columns([1, 1.5], gap="large")

    with lc2:
        st.markdown("""
        <div class="card">
          <div class="card-label">Point Lookup</div>
          <p style="font-size:0.85rem;color:#475569;line-height:1.7;margin:0 0 12px;">
            First upload a satellite image in the <b>Predict tab</b>.
            Then enter any latitude and longitude within Delhi NCT
            to find out whether that location is classified as
            Cool, Moderate, or Hot — and see what the surrounding
            area looks like.
          </p>
        </div>""", unsafe_allow_html=True)

        if "sel_lat" not in st.session_state:
            st.session_state["sel_lat"] = 28.6315
        if "sel_lon" not in st.session_state:
            st.session_state["sel_lon"] = 77.2167

        st.markdown("**Quick select a known location:**")
        presets = {
            "Connaught Pl": (28.6315, 77.2167),
            "Yamuna River":  (28.6600, 77.2800),
            "Ridge Forest":  (28.6900, 77.1300),
            "Okhla":         (28.5500, 77.2700),
            "IGI Airport":   (28.5562, 77.1000),
        }
        pcols = st.columns(len(presets))
        for i,(nm,(plat,plon)) in enumerate(presets.items()):
            if pcols[i].button(nm, key=f"pst{i}", use_container_width=True):
                st.session_state["sel_lat"] = plat
                st.session_state["sel_lon"] = plon
                st.rerun()

        ca, cb = st.columns(2)
        lat = ca.number_input(
            "Latitude (North)", min_value=28.40, max_value=28.88,
            value=float(st.session_state["sel_lat"]),
            step=0.001, format="%.4f", key="lat_in",
            label_visibility="visible")
        lon = cb.number_input(
            "Longitude (East)", min_value=76.84, max_value=77.35,
            value=float(st.session_state["sel_lon"]),
            step=0.001, format="%.4f", key="lon_in",
            label_visibility="visible")

        go = st.button("Find UHI Class at This Location",
                       type="primary", use_container_width=True)

        st.markdown("""
        <div class="card" style="margin-top:14px;">
          <div class="card-label">Reference Locations</div>
          <table>
            <tr><th>Location</th><th>Lat</th><th>Lon</th><th>Expected</th></tr>
            <tr><td>Connaught Place</td><td>28.6315</td><td>77.2167</td><td>Hot</td></tr>
            <tr><td>Yamuna Floodplain</td><td>28.6600</td><td>77.2800</td><td>Cool</td></tr>
            <tr><td>Ridge Forest</td><td>28.6900</td><td>77.1300</td><td>Cool</td></tr>
            <tr><td>Okhla Industrial</td><td>28.5500</td><td>77.2700</td><td>Hot</td></tr>
            <tr><td>Lodhi Garden</td><td>28.5928</td><td>77.2197</td><td>Cool</td></tr>
            <tr><td>Shahdara</td><td>28.6700</td><td>77.2900</td><td>Hot</td></tr>
            <tr><td>IGI Airport</td><td>28.5562</td><td>77.1000</td><td>Moderate</td></tr>
          </table>
        </div>""", unsafe_allow_html=True)

    with rc2:
        if not go:
            st.markdown("""
            <div class="empty-state">
              <div class="empty-icon">&#x1F4CD;</div>
              <div class="empty-title">Enter coordinates to look up</div>
              <div class="empty-sub">
                Select a preset location or type in a latitude and
                longitude, then click the button to find the UHI
                classification for that exact point.
              </div>
            </div>""", unsafe_allow_html=True)
        elif "label_map" not in st.session_state:
            st.warning("Please upload a satellite image in the Predict tab first.")
        else:
            lm   = st.session_state["label_map"]
            bnds = st.session_state["bounds"]
            H_   = st.session_state["H"]
            W_   = st.session_state["W"]
            row, col = latlon_to_px(lat, lon, bnds, H_, W_)

            if row is None:
                st.error("These coordinates fall outside the uploaded image extent.")
            else:
                cls = int(lm[row, col])
                if cls == -1:
                    st.warning("This pixel has no data — it may be cloud-masked "
                               "or outside Delhi's boundary.")
                else:
                    st.markdown(f"""
                    <div class="rbox {R_CSS[cls]}">
                      <div style="margin-bottom:6px;">
                        <span class="r-indicator"
                          style="background:{R_IND[cls]};"></span>
                        <span style="font-size:0.8rem;font-weight:600;
                          color:{R_IND[cls]};letter-spacing:0.08em;
                          text-transform:uppercase;">
                          {CLASSES[cls]} Zone
                        </span>
                      </div>
                      <div class="r-class {R_CLS[cls]}">
                        {lat:.4f}° N, {lon:.4f}° E
                      </div>
                      <div class="r-desc {R_CLS[cls]}">{DESC[cls]}</div>
                    </div>""", unsafe_allow_html=True)

                    d1, d2 = st.columns(2)
                    d1.markdown(f"""| | |
|---|---|
| Latitude | {lat:.4f}° N |
| Longitude | {lon:.4f}° E |
| Pixel location | row {row}, col {col} |
| UHI Class | {CLASSES[cls]} |""")
                    d2.markdown(f"""| | |
|---|---|
| Model | XGBoost Stage 5 |
| Accuracy | 94.01% |
| Features used | 53 spatial |
| Imagery | Landsat 9, 2023 |""")

                    # neighbourhood
                    r0,r1 = max(0,row-2), min(H_,row+3)
                    c0,c1 = max(0,col-2), min(W_,col+3)
                    vp = lm[r0:r1, c0:c1]; vp = vp[vp>=0]
                    if len(vp):
                        pc = int((vp==0).sum())/len(vp)*100
                        pm = int((vp==1).sum())/len(vp)*100
                        ph = int((vp==2).sum())/len(vp)*100
                        st.markdown("**Surrounding area (5x5 pixels, ~150 m radius)**")
                        st.markdown(f"""
<div class="card">
  <div style="display:flex;justify-content:space-between;
    align-items:center;margin-bottom:5px;">
    <span style="font-size:0.83rem;font-weight:600;color:#3B82F6;">Cool</span>
    <span style="font-size:0.83rem;color:#94A3B8;">{pc:.0f}%</span>
  </div>
  <div class="pbar-wrap">
    <div class="pbar-fill" style="width:{pc}%;background:#3B82F6;"></div>
  </div>
  <div style="display:flex;justify-content:space-between;
    align-items:center;margin:10px 0 5px;">
    <span style="font-size:0.83rem;font-weight:600;color:#D97706;">Moderate</span>
    <span style="font-size:0.83rem;color:#94A3B8;">{pm:.0f}%</span>
  </div>
  <div class="pbar-wrap">
    <div class="pbar-fill" style="width:{pm}%;background:#D97706;"></div>
  </div>
  <div style="display:flex;justify-content:space-between;
    align-items:center;margin:10px 0 5px;">
    <span style="font-size:0.83rem;font-weight:600;color:#EF4444;">Hot</span>
    <span style="font-size:0.83rem;color:#94A3B8;">{ph:.0f}%</span>
  </div>
  <div class="pbar-wrap">
    <div class="pbar-fill" style="width:{ph}%;background:#EF4444;"></div>
  </div>
</div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# TAB 3 — GEE GUIDE
# ═══════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div class="card">
      <div class="card-label">How to export your satellite image from Google Earth Engine</div>
      <p style="font-size:0.88rem;color:#475569;line-height:1.7;margin:0;">
        Google Earth Engine (GEE) is a free cloud platform for satellite imagery analysis.
        Follow the steps below to export a Landsat 9 image of Delhi that works directly
        with this application.
      </p>
    </div>""", unsafe_allow_html=True)

    cg1, cg2 = st.columns(2, gap="large")

    with cg1:
        st.markdown("""
        <div class="gee-step">
          <h4>Step 1 — Open Google Earth Engine</h4>
          <p>Go to <a href="https://code.earthengine.google.com"
          target="_blank">code.earthengine.google.com</a> and sign in
          with your Google account. Click <b>New Script</b>.</p>
        </div>
        <div class="gee-step">
          <h4>Step 2 — Paste the script</h4>
          <p>Copy the complete script from the right panel and paste it
          into the GEE code editor. Then click <b>Run</b>.</p>
        </div>
        <div class="gee-step">
          <h4>Step 3 — Start the export task</h4>
          <p>After running, go to the <b>Tasks</b> tab on the right panel.
          Click <b>RUN</b> next to <code>Delhi_L9_2023</code>.
          The export takes 5–15 minutes.</p>
        </div>
        <div class="gee-step">
          <h4>Step 4 — Download from Google Drive</h4>
          <p>Open <a href="https://drive.google.com" target="_blank">
          Google Drive</a>. Find <code>Delhi_L9_2023.tif</code>
          in the GEE_Exports folder and download it to your computer.</p>
        </div>
        <div class="gee-step">
          <h4>Step 5 — Upload to ThermaLens</h4>
          <p>Go to the <b>Predict tab</b> and upload the downloaded file.
          The model will classify it in under 15 seconds.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card" style="margin-top:12px;">
          <div class="card-label">Important Notes</div>
          <p style="font-size:0.82rem;color:#475569;line-height:1.9;margin:0;">
            Use <b>Collection 2, Tier 1, Level 2</b> (surface reflectance)<br>
            Export in <b>EPSG:4326</b> for correct coordinate display<br>
            Keep scale at <b>30 metres</b> (native Landsat resolution)<br>
            Summer months (May–August) give the strongest UHI contrast<br>
            Cloud cover filter below <b>10%</b> gives the cleanest results<br>
            Expected file size: <b>50–100 MB</b>
          </p>
        </div>""", unsafe_allow_html=True)

    with cg2:
        st.markdown("**GEE Export Script — copy and paste this exactly:**")
        st.code("""// ══════════════════════════════════════════
// ThermaLens — Landsat 9 Export Script
// Paste into GEE Code Editor and click Run
// ══════════════════════════════════════════

// Define Delhi NCT boundary
var delhi = ee.Geometry.Rectangle([76.84, 28.40, 77.35, 28.88]);

// Load Landsat 9 Collection 2 Level-2
var collection = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
  .filterDate("2023-05-01", "2023-08-31")
  .filterBounds(delhi)
  .filter(ee.Filter.lt("CLOUD_COVER", 10));

print("Scenes found:", collection.size());

// Create cloud-free median composite and clip to Delhi
var image = collection.median().clip(delhi);

// Select the 7 required bands in the correct order
var bands = image.select([
  "SR_B2",   // Blue
  "SR_B3",   // Green
  "SR_B4",   // Red
  "SR_B5",   // NIR
  "SR_B6",   // SWIR1
  "SR_B7",   // SWIR2
  "ST_B10"   // Thermal
]);

// Preview on map
Map.centerObject(delhi, 11);
Map.addLayer(bands,
  {bands: ["SR_B4", "SR_B3", "SR_B2"], min: 7000, max: 20000},
  "True Colour Composite");

// Export to Google Drive
Export.image.toDrive({
  image: bands,
  description: "Delhi_L9_2023",
  folder: "GEE_Exports",
  fileNamePrefix: "Delhi_L9_2023",
  scale: 30,
  region: delhi,
  crs: "EPSG:4326",
  fileFormat: "GeoTIFF",
  maxPixels: 1e9
});

print("Export task created. Go to Tasks tab and click RUN.");
""", language="javascript")

        st.markdown("""
        <div class="card" style="margin-top:12px;">
          <div class="card-label">Verify your file in Python</div>
        </div>""", unsafe_allow_html=True)
        st.code("""import rasterio
with rasterio.open("Delhi_L9_2023.tif") as src:
    print("Bands:", src.count)    # should be 7
    print("CRS:", src.crs)        # should be EPSG:4326
    print("Size:", src.shape)     # approx 1800 x 1700 pixels
    b2 = src.read(1)
    print("Band 2 range:", b2.min(), "to", b2.max())
    # Valid DN range is approximately 7000 to 50000
    # If you see all zeros or NaN, the wrong product was exported
""", language="python")

        st.info("""**Troubleshooting**

"No scenes found" — Widen the date range or raise the cloud cover filter to 20%.

"0 valid pixels" — Make sure you selected Collection 2 Level-2 (not Level-1).

"Wrong colours on map" — Confirm band order is B2, B3, B4, B5, B6, B7, B10.

"File too large" — Change scale from 30 to 60 in the export script.""")

# ═══════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════
st.markdown("""
<div class="footer">
  ThermaLens &nbsp;·&nbsp; Delhi Urban Heat Island Predictor &nbsp;·&nbsp;
  Landsat 9 &nbsp;·&nbsp; XGBoost Stage 5 &nbsp;·&nbsp;
  94.01% Accuracy &nbsp;·&nbsp; 2023
</div>""", unsafe_allow_html=True)