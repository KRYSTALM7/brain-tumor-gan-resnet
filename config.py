import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
TUMOR_DIR       = os.path.join(DATA_DIR, "yes")
NORMAL_DIR      = os.path.join(DATA_DIR, "no")
OUTPUT_DIR      = os.path.join(BASE_DIR, "outputs")
MODEL_DIR       = os.path.join(OUTPUT_DIR, "models")
PLOT_DIR        = os.path.join(OUTPUT_DIR, "plots")

# ── Image ──────────────────────────────────────────────────────────────────────
IMG_SIZE        = (256, 256)
IMG_CHANNELS    = 3

# ── Dataset ────────────────────────────────────────────────────────────────────
NUM_AUGMENTED   = 1000        # images per class after augmentation
TEST_SIZE       = 0.30
RANDOM_STATE    = 42

# ── GAN ────────────────────────────────────────────────────────────────────────
LATENT_DIM      = 100
GAN_EPOCHS      = 200
GAN_BATCH_SIZE  = 32
GAN_LR          = 0.0002
GAN_BETA_1      = 0.5

# ── Classifier ────────────────────────────────────────────────────────────────
CLF_EPOCHS      = 10
CLF_BATCH_SIZE  = 32
CLF_LR          = 0.0002
CLF_BETA_1      = 0.5
DROPOUT_RATE    = 0.4

# ── Labels ────────────────────────────────────────────────────────────────────
CLASS_NAMES     = {0: "No Tumor", 1: "Tumor"}

# ── Ensure output directories exist ───────────────────────────────────────────
for d in [OUTPUT_DIR, MODEL_DIR, PLOT_DIR]:
    os.makedirs(d, exist_ok=True)