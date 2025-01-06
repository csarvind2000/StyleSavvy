
# StyleSavvy: Multi-Modal AI-Powered Fashion Stylist Recommendation

StyleSavvy is an AI-powered fashion assistant designed to help users discover trendy outfit recommendations and find matching styles. The app utilizes advanced AI models, including OpenAI's GPT-4 and image search capabilities with ChromaDB, to provide personalized suggestions based on text or images.

## Features

- **Outfit Recommendations:** Get creative outfit ideas by simply asking a query in plain text.
- **Image Search:** Upload an outfit image to find similar styles from the dataset.
- **Quick Prompts:** Use pre-defined quick prompts for instant recommendations.
- **Image Prompts:** Select from sample outfit images to explore similar styles.

## How It Works

1. **Text-to-Image Search:** Enter a fashion-related query (e.g., "What can I wear with blue jeans?") and get outfit recommendations with matching images.
2. **Image-to-Image Search:** Upload an image of an outfit to find visually similar items.
3. **Quick Prompts:** Click on predefined prompts to explore popular styles.
4. **Image Prompts:** Use provided sample images to quickly search for matching styles.

## Tech Stack

- **Frontend:** [Gradio](https://gradio.app) for creating an interactive and responsive UI.
- **Backend:**
  - [OpenAI GPT-4](https://openai.com) for generating outfit recommendations.
  - [ChromaDB](https://chromadb.org) for managing and querying vector embeddings.
  - [Hugging Face Datasets](https://huggingface.co/datasets/jinaai/fashion-captions-de) for the fashion dataset.

## Installation

### Prerequisites

- Python 3.8+
- API keys for OpenAI and Hugging Face:
  - **OpenAI API Key:** [Get your key here](https://platform.openai.com/signup/).
  - **Hugging Face API Token:** [Get your token here](https://huggingface.co/settings/tokens).

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/stylesavvy.git
cd stylesavvy
```

### Step 2: Install Dependencies

Install the required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

### Step 3: Set Up Environment Variables

Create a `.env` file in the root directory and add the following:

```env
OPENAI_API_KEY=your_openai_api_key
HF_TOKEN=your_huggingface_api_token
```

### Step 4: Run the App

Start the Gradio app:

```bash
python stylesavvy2.py
```

The app will launch on `http://localhost:7860` in your web browser.

## Usage

### Text-to-Image Search

1. Enter a query in the "Get Outfit Recommendations" textbox (e.g., "What should I wear for a summer party?").
2. Click `Submit` to see recommendations and matching images.

### Image-to-Image Search

1. Upload an image of an outfit in the "Upload Outfit Image" section.
2. Click `Search Similar` to find matching styles.

### Quick Prompts

1. Click on any predefined prompt (e.g., "Summer casual outfits").
2. The app will display recommendations based on the selected prompt.

### Image Prompts

1. Select a sample image from the "Image Prompts" section.
2. Click the associated `Use Image` button to search for similar styles.

## Project Structure

```
.
├── stylesavvy2.py                # Main application file text-image & image-image RAG
├── stylesavvy.py                # Main application file text-image RAG
├── dataset/              # Folder for storing dataset images
├── data/                 # Folder for ChromaDB storage
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables
└── README.md             # Project documentation
```

## Dataset

The fashion dataset is sourced from [Hugging Face Datasets](https://huggingface.co/datasets/jinaai/fashion-captions-de). The dataset includes images and captions for a variety of fashion styles.

## Screenshots

### Main Interface
![Main Interface](https://via.placeholder.com/800x400)

### Recommendations
![Recommendations](https://via.placeholder.com/800x400)

## Future Enhancements

- Integration with live e-commerce platforms for real-time outfit purchases.
- Multi-language support for a global audience.
- Advanced filtering options for image searches.

## Acknowledgments

- [OpenAI](https://openai.com) for GPT-4.
- [Hugging Face](https://huggingface.co) for datasets.
- [Gradio](https://gradio.app) for the interactive UI framework.

## License

This project is licensed under the [MIT License](LICENSE).
