# 🧙 The Elder - Wisdom Guide LLM

**A philosophical AI trained on Bushido, Stoicism, and Native American wisdom traditions.**

The Elder is a conversational AI that provides guidance through Socratic dialogue, helping people develop character, find inner strength, and navigate life's challenges with wisdom and integrity.

## 🎯 Philosophy

The Elder draws from three complementary wisdom traditions:

### Bushido (武士道)
The Way of the Warrior - principles of honor, courage, compassion, respect, sincerity, honor, and loyalty.

### Stoicism
Teachings of Marcus Aurelius, Seneca, and Epictetus on self-mastery, acceptance, and living according to nature and reason.

### Native American Wisdom
Understanding of interconnection, balance with nature, seven generations thinking, and cyclical perspectives on life.

## 🌟 Features

- ✅ **Socratic Dialogue**: Teaches through questions, not lectures
- ✅ **Character Development**: Focuses on integrity, courage, discipline, and compassion
- ✅ **Practical Wisdom**: Applicable guidance for daily challenges
- ✅ **Non-Religious**: Universal spiritual principles without religious doctrine
- ✅ **Mobile-Optimized**: GGUF format for Android deployment
- ✅ **Open Source**: Free to use, modify, and share

## 📁 Project Structure

```
the-elder-llm/
├── data/
│   └── the_elder_dataset.jsonl    # 50+ wisdom training examples
├── configs/
│   ├── the_elder_system_prompt.txt
│   └── training_config.yaml
├── notebooks/
│   └── The_Elder_Training.ipynb   # Complete Colab training pipeline
├── scripts/
│   └── convert_to_gguf.sh         # GGUF conversion tool
├── releases/
│   └── The_Elder.gguf             # Mobile-ready model (created after training)
└── README.md
```

## 🚀 Quick Start Guide

### Option 1: Use Pre-Trained Model (Recommended)

**For immediate use:**

1. **Download the GGUF model:**
   ```bash
   # From Hugging Face (after training is complete)
   wget https://huggingface.co/Ishabdullah/The_Elder/resolve/main/The_Elder.gguf
   ```

2. **Install an Android LLM app:**
   - SmolChat
   - LM Studio Mobile
   - Pocket LLM
   - Any GGUF-compatible app

3. **Load the model:**
   - Place `The_Elder.gguf` in your app's model directory
   - Select it from the app's model list
   - Start conversing with The Elder!

### Option 2: Train Your Own

Follow the complete training pipeline to customize the model.

## 📚 Complete Training Pipeline

### Prerequisites

- Google account (for Colab)
- Hugging Face account with API token
- GitHub account with API token
- ~1 hour of time

### Step 1: Setup Tokens

**Hugging Face Token:**
1. Go to https://huggingface.co/settings/tokens
2. Create new token with **Write** permissions
3. Save as `HF_TOKEN`

**GitHub Token:**
1. Go to https://github.com/settings/tokens
2. Create classic token with `repo` and `workflow` scopes
3. Save as `GH_TOKEN`

### Step 2: Clone Repository

```bash
git clone https://github.com/Ishabdullah/the-elder-llm.git
cd the-elder-llm
```

### Step 3: Open Colab Notebook

1. Open: https://colab.research.google.com/github/Ishabdullah/the-elder-llm/blob/main/notebooks/The_Elder_Training.ipynb

2. Enable GPU:
   - Runtime → Change runtime type → T4 GPU

3. Add Secrets (🔑 icon):
   - `HF_TOKEN`: Your Hugging Face token
   - `GH_TOKEN`: Your GitHub token

4. Run all cells!

### Step 4: Monitor Training

The notebook will:
1. ✅ Setup environment and verify GPU
2. ✅ Clone your repository
3. ✅ Load the wisdom dataset (50+ examples)
4. ✅ Load TinyLlama base model with 4-bit quantization
5. ✅ Configure LoRA for parameter-efficient training
6. ✅ Train for 3 epochs (~30-45 minutes)
7. ✅ Test the trained model
8. ✅ Push to Hugging Face Hub
9. ✅ Convert to GGUF format for mobile
10. ✅ Upload GGUF file
11. ✅ Commit results to GitHub

### Step 5: Download & Deploy

After training completes:

```bash
# Download GGUF from Hugging Face
wget https://huggingface.co/Ishabdullah/The_Elder/resolve/main/The_Elder.gguf

# Or from GitHub releases
wget https://github.com/Ishabdullah/the-elder-llm/raw/main/releases/The_Elder.gguf
```

Transfer to your Android device and load in your preferred LLM app!

## 💬 Example Interactions

**User:** What is true strength?

**The Elder:** True strength is not in never falling, but in rising each time you fall. It is the courage to be vulnerable, to admit 'I don't know,' to ask for help. The oak tree appears strong, but it breaks in the hurricane. The willow bends and survives. Strength is gentleness with power held in reserve.

---

**User:** How should I respond when someone insults me?

**The Elder:** Consider: does this insult change who you are? The Stoics remind us that we cannot control the actions of others, only our response. Like a mountain that does not move when the wind howls, you remain unchanged by words. Respond with silence or with compassion, for anger is a fire that burns the one who carries it.

---

**User:** I'm afraid of failure. What should I do?

**The Elder:** Fear is natural, young one. But ask yourself: what is failure? Is it not simply another teacher? The warrior trains not to avoid falling, but to rise each time with greater wisdom. When you face what you fear, you discover it has no power over you - only the power you gave it in your mind.

## 🎓 Dataset Overview

The training dataset contains 50+ carefully crafted Q&A pairs covering:

- Character development and virtue
- Dealing with emotions (anger, fear, jealousy)
- Relationships and communication
- Purpose and meaning
- Death and impermanence
- Leadership and responsibility
- Discipline and self-mastery
- Suffering and resilience
- Balance and harmony
- And more...

Each response embodies:
- Wisdom from multiple traditions
- Socratic questioning technique
- Practical applicability
- Compassionate yet direct tone
- Metaphors from nature and life

## 🔧 Technical Details

### Base Model
- **TinyLlama-1.1B-Chat-v1.0**
- Chosen for efficiency on Colab free tier and mobile devices
- Can be substituted with Phi-2, Mistral-7B-instruct, or others

### Training Method
- **LoRA (Low-Rank Adaptation)**
- Rank: 16, Alpha: 32
- Target modules: all attention and MLP layers
- 4-bit quantization during training
- ~600MB LoRA adapter

### Training Configuration
- Epochs: 3
- Batch size: 4 (effective 16 with gradient accumulation)
- Learning rate: 2e-4 with cosine schedule
- Max sequence length: 512 tokens
- Training time: ~30-45 minutes on T4 GPU

### Output Formats
1. **Full Model** (~2.2GB): Standard Hugging Face format
2. **GGUF Q4_K_M** (~600MB): 4-bit quantized for mobile

## 📱 Mobile Deployment

### Recommended Apps for Android

1. **SmolChat**
   - Direct GGUF support
   - Clean interface
   - Free

2. **LM Studio Mobile**
   - Advanced features
   - Model management
   - Chat history

3. **Pocket LLM**
   - Lightweight
   - Fast inference
   - Multiple model support

### Installation Steps

1. Download `The_Elder.gguf`
2. Transfer to phone via USB or cloud
3. Open your LLM app
4. Add model from file
5. Start chatting!

### Performance

On typical Android devices:
- **Load time:** 2-5 seconds
- **Response time:** 5-15 seconds per message
- **Memory usage:** ~800MB RAM
- **Battery:** Moderate (local processing, no internet needed)

## 🔄 Creating New Models

Want to train a different persona? Use this pipeline as a template:

### 1. Create New Dataset

```bash
cp data/the_elder_dataset.jsonl data/my_model_dataset.jsonl
# Edit with your own Q&A pairs
```

### 2. Modify System Prompt

```bash
cp configs/the_elder_system_prompt.txt configs/my_model_system_prompt.txt
# Define your model's persona and knowledge
```

### 3. Update Configuration

Edit `configs/training_config.yaml`:
```yaml
model_card:
  model_name: "My_Model_Name"
  description: "Your model description"
```

### 4. Run Training

Update the Colab notebook with your new paths and run all cells!

This modular structure lets you create multiple specialized models:
- **The Strategist**: Military strategy and tactical thinking
- **The Healer**: Emotional support and trauma recovery
- **The Scientist**: Scientific thinking and skepticism
- **The Artist**: Creative process and aesthetic philosophy
- etc.

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- **Dataset expansion**: More diverse wisdom examples
- **Additional philosophies**: Buddhism, Taoism, etc.
- **Multilingual support**: Translate to other languages
- **Evaluation metrics**: Wisdom quality assessment
- **Fine-tuning on larger models**: Mistral-7B, Llama-2-7B

## 📄 License

**Apache 2.0** - Free to use, modify, and distribute with attribution.

## 🙏 Acknowledgments

Inspired by:
- The Bushido code and samurai philosophy
- Marcus Aurelius, Seneca, and Epictetus
- Native American wisdom keepers
- All who seek wisdom and share it freely

## 📞 Support & Contact

- **Issues**: https://github.com/Ishabdullah/the-elder-llm/issues
- **Discussions**: https://github.com/Ishabdullah/the-elder-llm/discussions
- **Hugging Face**: https://huggingface.co/Ishabdullah/The_Elder

## 🔗 Links

- **Colab Notebook**: [Open in Colab](https://colab.research.google.com/github/Ishabdullah/the-elder-llm/blob/main/notebooks/The_Elder_Training.ipynb)
- **Hugging Face Model**: https://huggingface.co/Ishabdullah/The_Elder
- **GGUF Download**: https://huggingface.co/Ishabdullah/The_Elder/resolve/main/The_Elder.gguf

---

## 🎯 Roadmap

### v1.0 (Current)
- ✅ Core wisdom dataset (50+ examples)
- ✅ LoRA fine-tuning pipeline
- ✅ GGUF mobile format
- ✅ Complete documentation

### v1.1 (Planned)
- ⏳ Expand dataset to 200+ examples
- ⏳ Add evaluation benchmarks
- ⏳ Multi-turn conversation improvements
- ⏳ Community wisdom submissions

### v2.0 (Future)
- ⏳ Larger base model (7B parameters)
- ⏳ Multilingual support
- ⏳ Voice interface
- ⏳ Companion mobile app

---

**"The warrior trains not to avoid falling, but to rise each time with greater wisdom."** - The Elder

*May this model serve as a guide on your path to wisdom and character.*
