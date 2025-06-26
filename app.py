import os
from flask import Flask, render_template, request
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer, M2M100ForConditionalGeneration, M2M100Tokenizer
from PIL import Image
import torch
from gtts import gTTS

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load captioning model
caption_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
caption_feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
caption_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Load translation model
translation_model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
translation_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

# Supported languages (without Hindi)
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

def generate_caption(image_path):
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert(mode="RGB")
    pixel_values = caption_feature_extractor(images=[image], return_tensors="pt").pixel_values
    output_ids = caption_model.generate(pixel_values, max_length=16)

    caption = caption_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption

def translate_caption(caption, target_lang_code):
    translation_tokenizer.src_lang = "en"
    encoded = translation_tokenizer(caption, return_tensors="pt")
    generated_tokens = translation_model.generate(**encoded, forced_bos_token_id=translation_tokenizer.get_lang_id(target_lang_code))
    translated = translation_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return translated

def speak_caption(text, lang_code, output_path="static/speech.mp3"):
    try:
        tts = gTTS(text=text, lang=lang_code)
        tts.save(output_path)
        return output_path
    except Exception as e:
        print("TTS Error:", e)
        return None

@app.route('/')
def index():
    return render_template('index.html', languages=LANGUAGES)

@app.route('/generate', methods=['POST'])
def generate():
    image_file = request.files['image']
    target_language = request.form['language']
    lang_code = LANGUAGES[target_language]

    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
    image_file.save(image_path)

    english_caption = generate_caption(image_path)
    final_caption = english_caption if lang_code == "en" else translate_caption(english_caption, lang_code)

    # TTS file
    tts_path = speak_caption(final_caption, lang_code)

    return render_template(
        'caption.html',
        caption=final_caption,
        image_path=image_path,
        audio_path=tts_path,
        language=target_language
    )

if __name__ == '__main__':
    app.run(debug=True)