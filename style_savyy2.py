import gradio as gr
from PIL import Image
from datasets import load_dataset
import os
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import numpy as np
import openai
from dotenv import load_dotenv
from huggingface_hub import login

# === STEP 1: Setup Environment and Hugging Face Login ===
load_dotenv()  # Load .env file for environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

# Hugging Face Login
HF_TOKEN = os.getenv("HF_TOKEN")  # Add your token to the .env file
if HF_TOKEN:
    login(HF_TOKEN)
else:
    print("Hugging Face token not found. Please set 'HF_TOKEN' in your .env file.")

# === STEP 2: Dataset and ChromaDB Setup ===
dataset_folder = "./dataset/fashion"
os.makedirs(dataset_folder, exist_ok=True)

# Load the dataset
print("Downloading the dataset from Hugging Face...")
ds = load_dataset("jinaai/fashion-captions-de")
if not os.listdir(dataset_folder):
    print("Saving dataset images...")
    for i in range(500):  # Save the first 500 images
        image = ds["train"][i]["image"]
        image.save(os.path.join(dataset_folder, f"fashion_{i+1}.png"))
    print("Images saved!")

# Setup ChromaDB
chroma_client = chromadb.PersistentClient(path="./data/fashion.db")
embedding_function = OpenCLIPEmbeddingFunction()
image_loader = ImageLoader()  # Initialize the ImageLoader

# Create ChromaDB collection with the ImageLoader
fashion_collection = chroma_client.get_or_create_collection(
    "fashion_collection", embedding_function=embedding_function, data_loader=image_loader
)

# Add images to ChromaDB
if fashion_collection.count() == 0:
    ids, uris = [], []
    for i, filename in enumerate(sorted(os.listdir(dataset_folder))):
        if filename.endswith(".png"):  # Ensure only image files are processed
            file_path = os.path.join(dataset_folder, filename)
            ids.append(str(i))
            uris.append(file_path)

    # Add images using URIs and IDs
    try:
        fashion_collection.add(ids=ids, uris=uris)
        print("Images added to ChromaDB.")
    except Exception as e:
        print(f"Error adding images to ChromaDB: {e}")

# === STEP 3: Backend Logic for Search and Recommendations ===

def generate_recommendations(user_query):
    """Generate outfit recommendations using GPT-4."""
    prompt = f"""
    You are a fashion expert. Provide creative outfit recommendations for the following query:
    "{user_query}"
    Use markdown formatting for better presentation.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a fashion expert."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating recommendations: {e}"

def text_to_image_search_with_gpt(query):
    """Perform text-to-image search and get GPT-4 recommendations."""
    try:
        # Perform text-to-image search
        results = fashion_collection.query(
            query_texts=[query], n_results=3, include=["uris"]
        )
        if not results["uris"]:
            return "No matching results found.", None, None, None

        # Get GPT-generated recommendations
        recommendations_text = generate_recommendations(query)

        # Extract top 3 images
        uris = results["uris"][0]
        images = [Image.open(uri) for uri in uris]

        return recommendations_text, images[0], images[1] if len(images) > 1 else None, images[2] if len(images) > 2 else None
    except Exception as e:
        return f"Error: {e}", None, None, None

def image_to_image_search(image):
    """Perform image-to-image search."""
    try:
        with Image.open(image) as img:
            img = img.convert('RGB').resize((224, 224))
            img_array = np.array(img)
        embedding = embedding_function([img_array])
        results = fashion_collection.query(
            query_embeddings=embedding, n_results=3, include=["uris"]
        )
        if not results["uris"]:
            return "No similar items found.", None, None, None

        # Extract top 3 images
        uris = results["uris"][0]
        images = [Image.open(uri) for uri in uris]

        return "Here are similar items:", images[0], images[1] if len(images) > 1 else None, images[2] if len(images) > 2 else None
    except Exception as e:
        return f"Error: {e}", None, None, None

with gr.Blocks(css="""
    .gradio-container {
        background-color: #0D1B2A;  /* Dark blue background */
        font-family: 'Arial', sans-serif;
        color: #FFFFFF;  /* White text for content */
        min-height: 100vh;
    }
    .gr-button {
        background-color: #1B263B !important;  /* Slightly lighter blue for buttons */
        color: #FFFFFF !important;
        font-weight: bold;
        border-radius: 5px;
        padding: 10px;
        border: 2px solid #415A77;  /* Subtle border for buttons */
    }
    .gr-prompt-button {
        border: 2px solid #415A77; 
        background: transparent; 
        color: #FFFFFF; 
        font-weight: bold; 
        cursor: pointer; 
        padding: 8px 15px; 
        border-radius: 5px; 
        margin-right: 10px;
    }
    .gr-textbox-label {
        font-weight: bold;
        color: #E0E1DD;  /* Softer white for labels */
    }
    .gr-image-label {
        color: #E0E1DD;
        font-size: 1.1rem;
        font-weight: bold;
    }
    .gr-textbox {
        background-color: #1B263B !important;  /* Dark blue textbox background */
        color: #FFFFFF !important;  /* White text */
        border: 2px solid #415A77 !important;  /* Subtle border for the textbox */
        border-radius: 5px;
        padding: 10px;
    }
""") as app:
    gr.Markdown(
        """
        <div style="text-align: center; padding: 20px;">
            <h1 style="font-size: 3.5rem; color: #E0E1DD;">StyleSavvy</h1>
            <p style="font-size: 1.3rem; font-weight: bold; color: #E0E1DD;">
                Your Personal AI Fashion Stylist for Trendy Outfit Ideas
            </p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            # Input for text-to-image
            text_input = gr.Textbox(
                label="Get Outfit Recommendations",
                placeholder="e.g., What can I wear with blue jeans?",
                elem_classes=["gr-textbox-label", "gr-textbox"]
            )
            text_button = gr.Button("Submit")

            # Prompts below the input
            gr.Markdown("<p style='font-weight: bold; margin-top: 10px; color: #E0E1DD;'>Quick Prompts:</p>")
            with gr.Row():
                prompt1 = gr.Button("Summer casual outfits", elem_classes=["gr-prompt-button"])
                prompt2 = gr.Button("Office wear with beige pants", elem_classes=["gr-prompt-button"])
                prompt3 = gr.Button("What matches a leather jacket?", elem_classes=["gr-prompt-button"])

            # Input for image-to-image
            image_input = gr.Image(type="filepath", label="Upload Outfit Image")
            image_button = gr.Button("Search Similar")

            # Add Image Prompts for Image-to-Image Search
            gr.Markdown("<p style='font-weight: bold; margin-top: 20px; color: #E0E1DD;'>Image Prompts:</p>")
            with gr.Row():
                image_prompt1 = gr.Image(value="./input/Image1.png", interactive=False, label="Prompt 1")
                image_button1 = gr.Button("Use Image 1")
                image_prompt2 = gr.Image(value="./input/Image2.png", interactive=False, label="Prompt 2")
                image_button2 = gr.Button("Use Image 2")
                image_prompt3 = gr.Image(value="./input/Image3.png", interactive=False, label="Prompt 3")
                image_button3 = gr.Button("Use Image 3")

        with gr.Column(scale=3, min_width=900):
            with gr.Row():
                output_label = gr.Textbox(
                    label="Recommendations", lines=5, interactive=False, elem_classes=["gr-textbox-label", "gr-textbox"]
                )
            with gr.Row():
                output_image1 = gr.Image(label="Match 1", interactive=False, elem_classes=["gr-image-label"])
                output_image2 = gr.Image(label="Match 2", interactive=False, elem_classes=["gr-image-label"])
                output_image3 = gr.Image(label="Match 3", interactive=False, elem_classes=["gr-image-label"])

    # Define callbacks
    text_button.click(
        text_to_image_search_with_gpt,
        inputs=[text_input],
        outputs=[output_label, output_image1, output_image2, output_image3],
    )

    image_button.click(
        image_to_image_search,
        inputs=[image_input],
        outputs=[output_label, output_image1, output_image2, output_image3],
    )

    # Add Image Prompt button functionality
    image_button1.click(lambda: "./input/Image1.png", outputs=image_input)
    image_button2.click(lambda: "./input/Image2.png", outputs=image_input)
    image_button3.click(lambda: "./input/Image3.png", outputs=image_input)

    # Add prompt click functionality for text prompts
    prompt1.click(lambda: "Summer casual outfits", outputs=text_input)
    prompt2.click(lambda: "Office wear with beige pants", outputs=text_input)
    prompt3.click(lambda: "What matches a leather jacket?", outputs=text_input)

# Launch the app
app.launch()


