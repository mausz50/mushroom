import streamlit as st
from fastai.vision.all import load_learner, PILImage

st.set_page_config(page_title="Mushroom classifier", page_icon="🍄")

st.title("🍄 Welcome to my first trained model!")
st.write(
    "Upload a photo to try it. The model knows three mushrooms: "
    "one edible, one fun one and one deadly:"
)
st.markdown(
    "- **Amanita muscaria** (fly agaric)\n"
    "- **Amanita phalloides** (death cap)\n"
    "- **Russula xerampelina** (crab brittlegill)"
)
st.warning("⚠️ This is a learning project. Never use it to decide whether a mushroom is safe to eat.")


@st.cache_resource  # load the model once, not on every click
def get_model():
    return load_learner("export.pkl")


learn_inf = get_model()

uploaded = st.file_uploader("Select your mushroom", type=["jpg", "jpeg", "png", "webp"])

if uploaded is not None:
    img = PILImage.create(uploaded.getvalue())
    st.image(img.to_thumb(256, 256))
    pred, pred_idx, probs = learn_inf.predict(img)
    st.subheader(f"Prediction: {pred}")
    st.write(f"Probability: {probs[pred_idx]:.04f}")
