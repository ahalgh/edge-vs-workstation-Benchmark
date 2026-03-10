"""Standard prompt datasets for reproducible LLM benchmarking."""

# Short prompts (1-2 sentences) - test latency
SHORT_PROMPTS = [
    "What is the capital of France?",
    "Explain quantum computing in one sentence.",
    "Write a haiku about the ocean.",
    "What is 2 + 2?",
    "Name three primary colors.",
    "Define the word 'serendipity'.",
    "What year did World War II end?",
    "Translate 'hello' to Spanish.",
    "What is the speed of light?",
    "Name the largest planet in our solar system.",
]

# Medium prompts (paragraph-length) - test throughput
MEDIUM_PROMPTS = [
    "Explain the process of photosynthesis in detail, including the light-dependent and light-independent reactions. Describe the role of chlorophyll and the importance of this process for life on Earth.",
    "Describe the key differences between supervised learning, unsupervised learning, and reinforcement learning in machine learning. Provide an example use case for each approach.",
    "Write a detailed comparison of TCP and UDP protocols, explaining when each should be used, their advantages and disadvantages, and how they handle data transmission differently.",
    "Explain how a transformer neural network works, including the attention mechanism, positional encoding, and how it processes sequential data differently from recurrent neural networks.",
    "Describe the architecture of a modern GPU, including streaming multiprocessors, CUDA cores, tensor cores, memory hierarchy, and how these components work together for parallel computation.",
    "Explain the CAP theorem in distributed systems. Describe each property (Consistency, Availability, Partition tolerance) and why a system can only guarantee two of the three at any given time.",
    "Describe the process of training a large language model from scratch, including data collection, tokenization, pre-training objectives, fine-tuning, and alignment techniques like RLHF.",
    "Write a comprehensive overview of the RISC-V instruction set architecture, its design philosophy, advantages over proprietary ISAs, and its growing adoption in edge computing devices.",
    "Explain how public key cryptography works, including RSA key generation, encryption, decryption, and digital signatures. Describe why it is considered secure and its limitations.",
    "Describe the evolution of computer memory technology from magnetic core memory to modern HBM3, including DRAM, SRAM, flash memory, and the trade-offs between speed, density, and cost.",
]


def get_prompts(count: int = 10, prompt_type: str = "short") -> list[str]:
    """Get a list of standard prompts for benchmarking.

    Args:
        count: Number of prompts to return.
        prompt_type: "short" for latency testing, "medium" for throughput testing.
    """
    source = SHORT_PROMPTS if prompt_type == "short" else MEDIUM_PROMPTS
    # Cycle through prompts if count > available
    prompts = []
    while len(prompts) < count:
        prompts.extend(source)
    return prompts[:count]
