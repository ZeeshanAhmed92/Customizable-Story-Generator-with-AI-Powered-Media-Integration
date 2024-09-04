import streamlit as st
import os
import torch
from PIL import Image
from gtts import gTTS
from groq import Groq
from io import BytesIO
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from deep_translator import GoogleTranslator 
from diffusers.utils import export_to_video
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate
from diffusers import StableDiffusionPipeline,  DiffusionPipeline, DPMSolverMultistepScheduler

# Language options for translation
languages = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Chinese": "zh-CN",
    "Hindi": "hi",
    "Japanese": "ja",
    "Russian": "ru",
    "Portuguese": "pt"}

# Initialize Groq client with API key
client = Groq(api_key="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")

# Check the availability of cpu and gpu
device ="cpu"

# Initialize Stable Diffusion pipeline
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe = pipe.to(device)

# Initialize video generation pipeline
video_pipe = DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_576w", torch_dtype=torch.float32)
video_pipe.scheduler = DPMSolverMultistepScheduler.from_config(video_pipe.scheduler.config)
video_pipe = video_pipe.to(device)

# Define the header text with custom color using HTML
header_html1 = """
    <h1 style="color:#3c81ba;">Enchanted Narratives: Interactive Storytelling with AI</h1>
"""

# Render the HTML in Streamlit
st.markdown(header_html1, unsafe_allow_html=True)

image = Image.open("ai image.jpeg")
st.image(image, caption="Unleash your imagination with Enchanted Narratives – where AI brings your stories to life with interactive creativity and magical storytelling.")

# Translation options
st.markdown("<h2 style='color:#3c81ba;'>Select Input Language</h2>", unsafe_allow_html=True)
source_language = st.selectbox("Select the source language", list(languages.keys()))

def generate_video(prompt):
    # Convert the prompt to the appropriate type if necessary
    if isinstance(prompt, torch.Tensor):
        prompt = prompt.to(torch.float16)

    video_frames = video_pipe(prompt, num_inference_steps=40, height=320, width=576, num_frames=24).frames
    
    # Convert video frames back to float32 if needed by other parts of the code
    video_frames = [frame.to(torch.float32) for frame in video_frames]
    
    video_path = export_to_video(video_frames)
    return video_path


def translate_text(text, source_lang, target_lang):
    try:
        if source_lang != target_lang:
            translated = GoogleTranslator(source=languages[source_lang], target=languages[target_lang]).translate(text)
            return translated
        else:
            return text  # No translation needed
    except Exception as e:
        st.error(f"Translation error: {e}")
        return text  # Return original text if translation fails

def generate_story(prompt, max_length, lang='en'):
    # Truncate prompt if it exceeds a reasonable length based on language
    if lang != 'en':
        max_prompt_length = 1500  # Adjust this value based on API documentation for non-English
    else:
        max_prompt_length = 2000  # Adjust this value based on API documentation for English

    if len(prompt) > max_prompt_length:
        prompt = prompt[:max_prompt_length]

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt}
        ],
        model="llama3-8b-8192"
    )
    story = chat_completion.choices[0].message.content
    return story

def generate_image(prompt, guidance_scale=0.9, num_inference_steps=50, seed=42):
    generator = torch.Generator(device).manual_seed(seed)
    image = pipe(prompt, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps).images[0]
    return image

def display_images_for_story(story):
    paragraphs = story.split('\n\n')  # Assuming paragraphs are separated by double newlines
    for i, paragraph in enumerate(paragraphs):
        st.write(f"**Paragraph {i + 1}:**")
        st.write(paragraph)
        image = generate_image(paragraph)
        st.image(image, caption=f"Image for paragraph {i + 1}")

# Custom Prompt Option
st.markdown("<h2 style='color:#3c81ba;'>Use Custom Prompt and Plot Line</h2>", unsafe_allow_html=True)
use_custom_prompt = st.checkbox("")

# Disable other inputs when custom prompt is used
if use_custom_prompt:
    st.markdown("<h3 style='color:#3c81ba;'>Enter your custom prompt</h3>", unsafe_allow_html=True)
    placeholder_text = "Enter your custom prompt here in " + source_language
    custom_prompt = st.text_area("", placeholder=placeholder_text, key='custom_prompt1')

    st.markdown("<h3 style='color:#3c81ba;'>Enter key plot points</h3>", unsafe_allow_html=True)
    plot_points = st.text_area("", key='plot_line')

    st.markdown("<h3 style='color:#3c81ba;'>Select the story length</h3>", unsafe_allow_html=True)
    story_length = st.slider(label="", min_value=100, max_value=2000, value=500, step=100)

else:
    # Genre-specific roles
    genre_roles = {
        "Fantasy": ["Hero", "Villain", "Mentor", "Sidekick", "Magical Creature"],
        "Sci-Fi": ["Hero", "Scientist", "AI Companion", "Alien", "Rebel Leader"],
        "Mystery": ["Detective", "Suspect", "Victim", "Witness", "Accomplice"],
        "Romance": ["Protagonist", "Love Interest", "Best Friend", "Rival", "Matchmaker"]
    }

    # User inputs for the story
    st.markdown("<h3 style='color:#3c81ba;'>Select the genre of the Story</h3>", unsafe_allow_html=True)
    genre = st.selectbox("", list(genre_roles.keys()))

    st.markdown("<h3 style='color:#3c81ba;'>Number of Main Characters</h3>", unsafe_allow_html=True)
    num_characters = st.selectbox("", options=[1, 2, 3, 4, 5])

    characters = []

    for i in range(num_characters):
        st.markdown(f"**<span style='color:#3c81ba'>Name of Character {i+1}</span>**", unsafe_allow_html=True)
        name = st.text_input("", key=f"text_input_{i}")

        st.markdown(f"**<span style='color:#3c81ba'>Role of Character {i+1}</span>**", unsafe_allow_html=True)
        role = st.selectbox("", list(genre_roles[genre]), key=f"selectbox_input_{i}")

        characters.append({"name": name, "role": role})

    st.markdown("<h3 style='color:#3c81ba;'>Select the Setting of the Story</h3>", unsafe_allow_html=True)
    setting = st.selectbox("", ["Medieval Kingdom", "Futuristic City", "Small Town"])

    st.markdown("<h3 style='color:#3c81ba;'>Specify Key Plot Points or Events</h3>", unsafe_allow_html=True)
    plot_points = st.text_area("", key='plot_points')

    st.markdown("<h3 style='color:#3c81ba;'>Select the Length of the Story</h3>", unsafe_allow_html=True)
    story_length = st.radio("", options=[200, 400, 600, 800, 1000])

    st.markdown("<h3 style='color:#3c81ba;'>Select the Tone of the Story</h3>", unsafe_allow_html=True)
    tone = st.selectbox("", ["Humorous", "Dark", "Dramatic", "Mystery"])

    st.markdown("<h3 style='color:#3c81ba;'>Provide a Custom Prompt or Opening Line</h3>", unsafe_allow_html=True)
    custom_prompt = st.text_area("", key='custom_prompt')

    st.markdown("<h3 style='color:#3c81ba;'>Select the Dialogue Style</h3>", unsafe_allow_html=True)
    dialogue_style = st.selectbox("", ["Formal", "Casual", "Witty"])

    st.markdown("<h3 style='color:#3c81ba;'>Select the Narrative Voice</h3>", unsafe_allow_html=True)
    voice = st.selectbox("", ["First-Person", "Third-Person"])

# Declare the output format
st.markdown("<h3 style='color:#3c81ba;'>Select the Output Format</h3>", unsafe_allow_html=True)
output_format = st.selectbox("", ["Text on Screen", "Downloadable PDF", "Audio Narration", "Images", "Video"])

# Translation options
st.markdown("<h2 style='color:#3c81ba;'>Select Output Language</h2>", unsafe_allow_html=True)
target_language = st.selectbox("Select the target language", list(languages.keys()))

# Generate the story
if st.button("Generate Story"):
    if use_custom_prompt:
        prompt = custom_prompt + "\n" + (plot_points if isinstance(plot_points, str) else ' '.join(plot_points))
        prompt_language = source_language
    else:
        prompt = (
            f"Generate a {story_length} {genre} story set in {setting}.\n "
            f"The story should include {characters}.\n "
            f"The plot involves {plot_points}. \n"
            f"The tone should be {tone}.\n"
            f"The dialogue style should be {dialogue_style}.\n"
            f"The narrative voice should be {voice}.\n"
        )
        prompt_language = 'en'  # Default to English if not using a custom prompt

    story = generate_story(prompt, story_length, lang=prompt_language)

    # Translate the generated story
    if source_language != target_language:
        story = translate_text(story, source_language, target_language)

    if output_format == "Text on Screen":
        st.write(story)
        # if st.button("Generate Images for Each Paragraph"):
        #     display_images_for_story(story)
    elif output_format == "Downloadable PDF":
        # Display the generated story
        st.write(story)
        def create_pdf(story):
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            styles = getSampleStyleSheet()

            # Split the story into paragraphs
            paragraphs = story.split('\n\n')  # Assuming paragraphs are separated by double newlines

            # Create Paragraph objects for each split section
            content = [Paragraph(p, style=styles['Normal']) for p in paragraphs]

            doc.build(content)

            buffer.seek(0)
            return buffer

        pdf_buffer = create_pdf(story)
        st.download_button("Download PDF", pdf_buffer, "story.pdf")
        
        # if st.button("Generate Images for Each Paragraph"):
        #     display_images_for_story(story)

    elif output_format == "Audio Narration":
        # Display the generated story
        st.write(story)
        tts = gTTS(text=story, lang=languages[target_language])
        audio_buffer = BytesIO()
        tts.write_to_fp(audio_buffer)
        st.audio(audio_buffer, format='audio/mp3')

    elif output_format == "Images":
        # Display the generated story
        st.markdown("<h3 style='color:#3c81ba;'>Generating Images...</h3>", unsafe_allow_html=True)
        display_images_for_story(story)
        
    elif output_format == "Video":
        # Display the generated story
        st.write(story)
        st.markdown("<h3 style='color:#3c81ba;'>Generating Videos...</h3>", unsafe_allow_html=True)
        paragraphs = story.split('\n\n')  # Assuming paragraphs are separated by double newlines
        for i, paragraph in enumerate(paragraphs):
            st.write(f"**Paragraph {i + 1}:**")
            st.write(paragraph)
            video_path = generate_video(paragraph)
            st.video(video_path)
