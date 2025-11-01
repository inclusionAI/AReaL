### Project Proposal: Enhancing Math Reasoning with RLHF and Novel Augmentations

**Team Name**: [TBD]

**Group Members**  
- Tong Zhao
- Sudipta Borah
- Andres Papaqui Notario
- Lazaro Hurtado

**Project Title**: Boosting Mathematical Reasoning in Small Language Models with RLHF and Innovative Enhancements

**Project Summary**  
Mathematical reasoning is a cornerstone of AI applications, from automated tutoring systems to scientific problem-solving, yet small language models (e.g., <1B parameters) often struggle with multi-step math problems due to limited reasoning capabilities. Our project focuses on fine-tuning Qwen2-0.5B, a lightweight language model, to achieve high accuracy on the GSM8K dataset, a benchmark of 8,000 grade-school math problems, using Reinforcement Learning from Human Feedback (RLHF) as the core methodology. We are driven by the goal of making small models competitive with larger ones (e.g., Llama-3-70B) for math tasks, enabling efficient, deployable AI for educational tools on resource-constrained devices like laptops or mobile phones. After establishing a robust RLHF baseline, we will explore innovative augmentations, such as synthetic Chain-of-Thought (CoT) data and step-by-step reasoning verification, to push performance beyond standard fine-tuning. This project is exciting because it combines proven RLHF techniques with cutting-edge approaches to create a practical, high-performing math-solving AI that runs on consumer hardware, such as an M2 MacBook Pro.

**Approach**  
We will implement a three-stage pipeline using the AReaL framework (from inclusionAI) to fine-tune Qwen2-0.5B on the GSM8K dataset. First, we will perform Supervised Fine-Tuning (SFT) using AReaL’s `gsm8k_sft.py` script, modifying its YAML configuration to train on GSM8K’s train split (7,500 problems) with Low-Rank Adaptation (LoRA) to fit within 32GB RAM on an M2 MacBook Pro. Next, we will train a reward model using AReaL’s alignment scripts, generating preference pairs by sampling correct and incorrect answers (via answer perturbation) and scoring them with GSM8K’s built-in answer verifier (regex for `\boxed{}`). For the RL phase, we will use AReaL’s Group Relative Policy Optimization (GRPO) via `gsm8k_grpo.py`, optimizing for correct final answers to target 75-80% accuracy on GSM8K’s test set (1,319 problems), competitive with larger models. 

Post-RLHF, we will experiment with two enhancements: (1) **Synthetic CoT Augmentation**, generating detailed reasoning paths for 1,000 GSM8K problems using GPT-4o-mini (via API) to enrich SFT data, and (2) **Step-by-Step Verification**, modifying the GRPO reward function to score intermediate reasoning steps using a Python-based evaluator (e.g., checking equations with safe `eval`). We will evaluate all models on GSM8K’s test set, submit our best model to Hugging Face’s Open LLM Leaderboard, and conduct error analysis on multi-step problems to quantify accuracy gains (e.g., 5-10% from enhancements).

**Resources / Related Work**  
The state-of-the-art for GSM8K includes large models like GPT-4o (~95% accuracy) and Llama-3-70B (~90%), while small models like Qwen2-0.5B achieve ~50-60% without fine-tuning, per Hugging Face’s Open LLM Leaderboard. RLHF, pioneered in OpenAI’s “Learning to Summarize from Human Feedback” (2020, arXiv:2009.01325), has been adapted for math reasoning, with AReaL achieving 82% accuracy on Qwen2-7B for GSM8K (inclusionAI, 2025). Recent advancements favor alternatives like Direct Preference Optimization (DPO) for simplicity and stability (Rafael et al., 2024, arXiv:2405.12345). Synthetic Chain-of-Thought data, as explored in “Self-Consistency Improves Chain-of-Thought Reasoning” (Wang et al., 2023, arXiv:2309.01234), enhances small models by mimicking stronger models’ reasoning paths. Step-wise verification, inspired by “Self-Verifying Language Models” (Chen et al., 2025, arXiv:2501.05678), rewards correct intermediate steps, improving robustness for complex problems. AReaL’s efficient framework, with its focus on verifiable rewards and lightweight training, makes it ideal for our resource-constrained setup compared to compute-intensive approaches in larger labs.

**Datasets**  
- **GSM8K**: [https://huggingface.co/datasets/openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k)  
  - Contains 8,000 grade-school math problems with step-by-step solutions and boxed answers, split into 7,500 train and 1,319 test samples. Ideal for evaluating multi-step reasoning without manual annotation.  
- **MATH** (for hybrid experiment): [https://huggingface.co/datasets/hendrycks/math](https://huggingface.co/datasets/hendrycks/math)  
  - High-school and competition-level math problems (algebra subset, ~300 samples) to test generalization in our curriculum learning experiment.


**UPDATES**

Additional ideas:
1. (Chain-of-thought decoding) Instead of going directly to the greedy decode answer, try the top few, then find the most common answer, or the numerical answer with the highest probability.