# Text Diffusion Visualizer

This project demonstrates text diffusion with backmasking, showing how diffusion models gradually transform noise into coherent text. The UI visualizes the diffusion process in real-time, including the backmasking steps.

## Architecture

The project consists of two parts:

1. **Flask Backend**: Handles the text diffusion process using the `generate_backmasking.py` module
2. **Next.js Frontend**: Provides a user interface to interact with and visualize the diffusion process

## Prerequisites

- Node.js (v18 or higher)
- Python 3.8 or higher
- Required Python packages (listed in requirements.txt)
- Appropriate model weights (check the Flask backend for model details)

## Setup and Running

### Mock Mode (No Backend Required)

For development and UI testing, the application includes a "Mock Mode" toggle that allows you to use the UI with simulated data:

1. Install dependencies:
   ```bash
   cd nextjs-app
   npm install
   ```

2. Start the development server:
   ```bash
   npm run dev
   ```

3. Toggle on "Mock Mode" in the UI header to work without a backend server

This mode simulates the text diffusion process including backmasking steps, block scores, and visualizations without requiring the heavyweight ML model to be loaded.

### Flask Backend (for Real Processing)

If you want to use the real processing capabilities, you need to run the Flask backend:

1. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

2. Install the required packages:
   ```bash
   pip install flask flask-cors torch numpy transformers
   ```

3. Start the Flask server:
   ```bash
   python app.py
   ```

The backend will be available at http://localhost:5000.

### Next.js Frontend

1. Install dependencies:
   ```bash
   cd nextjs-app
   npm install
   ```

2. Start the development server:
   ```bash
   npm run dev
   ```

The frontend will be available at http://localhost:3000.

3. Make sure "Mock Mode" is disabled to use the real backend

## Features

- Enter text prompts for the diffusion model to respond to
- Visualize the diffusion process in real-time, including:
  - Progressive demasking of tokens
  - Backmasking steps where lower-confidence tokens are masked again
  - Score metrics for each block of generated text
- Customize diffusion parameters like temperature, backmasking intensity, etc.
- Switch between mock mode (simulated data) and real backend processing

## How It Works

1. **Text Diffusion**: The model starts with a fully masked sequence and gradually reveals tokens until the generation is complete.
2. **Backmasking**: During generation, blocks with poor quality scores are partially masked again, allowing the model to reconsider and potentially improve those sections.
3. **Visualization**: The UI shows the diffusion process in real-time, highlighting masked and unmasked tokens.

## Customizing Parameters

The advanced parameters section allows you to customize:

- `steps`: Number of diffusion steps
- `gen_length`: Total length of generated text
- `block_length`: Size of text blocks for backmasking
- `temperature`: Sampling temperature (higher = more random)
- `backmasking_alpha`: Controls steepness of backmasking probability curve
- `backmasking_intensity`: Overall intensity of backmasking (0-1)
- `backmasking_frequency`: Apply backmasking every N blocks
- `backmasking_threshold`: Quality threshold below which backmasking is triggered

This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
