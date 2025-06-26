import os
import streamlit as st
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer, M2M100ForConditionalGeneration, M2M100Tokenizer
from gtts import gTTS
import torch

# Setup upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load models
caption_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
caption_feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
caption_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

translation_model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
translation_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

# Supported languages
LANGUAGES = {
    "English": "en",
    "French": "fr",
    "Spanish": "es",
    "German": "de",
    "Italian": "it",
    "Japanese": "ja",
    "Korean": "ko",
    "Chinese": "zh",
    "Russian": "ru",
    "Arabic": "ar"
}

# Functions
def generate_caption(image):
    if image.mode != "RGB":
        image = image.convert(mode="RGB")
    pixel_values = caption_feature_extractor(images=[image], return_tensors="pt").pixel_values
    output_ids = caption_model.generate(pixel_values, max_length=16)  # GPT2 doesn't support beam search
    caption = caption_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption

def translate_caption(caption, target_lang_code):
    translation_tokenizer.src_lang = "en"
    encoded = translation_tokenizer(caption, return_tensors="pt")
    generated_tokens = translation_model.generate(**encoded, forced_bos_token_id=translation_tokenizer.get_lang_id(target_lang_code))
    translated = translation_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return translated

def speak_caption(text, lang_code, path="speech.mp3"):
    try:
        tts = gTTS(text=text, lang=lang_code)
        tts.save(path)
        return path
    except Exception as e:
        st.error(f"TTS Error: {e}")
        return None

# Streamlit UI
st.set_page_config(page_title="Multilingual Caption Generator", layout="centered")
st.title("🖼️ Multilingual Image Caption Generator with Speech 🎤")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
language = st.selectbox("Select Language", list(LANGUAGES.keys()))

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Generate Caption"):
        with st.spinner("Generating caption..."):
            caption = generate_caption(image)
            lang_code = LANGUAGES[language]
            final_caption = caption if lang_code == "en" else translate_caption(caption, lang_code)
            st.success(f"Caption ({language}): {final_caption}")
            
            audio_path = speak_caption(final_caption, lang_code)
            if audio_path:
                st.audio(audio_path)
