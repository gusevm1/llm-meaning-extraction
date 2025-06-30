"""
Dataset of sentences for semantic compression/decompression testing.
Contains longer, information-rich sentences with varied complexity.
"""

COMPRESSION_DATASET = [
    "The ancient Mediterranean trading routes, established by Phoenician merchants around 1200 BCE, fundamentally transformed economic relationships between distant civilizations by enabling the exchange of not only goods like purple dye, cedar wood, and precious metals, but also ideas, technologies, and cultural practices that would shape the development of Western civilization.",
    
    "Climate scientists at the International Panel on Climate Change have determined through extensive analysis of ice core samples, satellite temperature measurements, and oceanic thermal data that global average temperatures have risen by approximately 1.1 degrees Celsius since pre-industrial times, with the most dramatic increases occurring in the Arctic regions where permafrost melting is accelerating at unprecedented rates.",
    
    "The human brain's neuroplasticity allows for remarkable adaptation throughout life, as demonstrated by stroke patients who recover speech and motor functions through intensive rehabilitation therapy that essentially rewires neural pathways, creating new connections that bypass damaged areas and restore cognitive abilities previously thought to be permanently lost.",
    
    "Modern cryptocurrency blockchain technology utilizes distributed ledger systems with cryptographic hash functions to create immutable transaction records that are verified by thousands of independent nodes worldwide, eliminating the need for traditional banking intermediaries while consuming enormous amounts of electrical energy in the process.",
    
    "The James Webb Space Telescope, positioned at the second Lagrange point approximately 1.5 million kilometers from Earth, uses its 6.5-meter segmented mirror and advanced infrared instruments to observe the universe's earliest galaxies, some formed just 400 million years after the Big Bang, providing unprecedented insights into cosmic evolution and star formation processes.",
    
    "Quantum entanglement, described by Einstein as 'spooky action at a distance,' occurs when two particles become correlated in such a way that measuring the quantum state of one particle instantly determines the state of its partner, regardless of the physical distance separating them, challenging our classical understanding of locality and causality in physics.",
    
    "The Amazon rainforest ecosystem supports over 400 billion individual trees representing approximately 16,000 different species, creates roughly 20% of the world's oxygen, and serves as home to an estimated 10% of Earth's biodiversity, while also playing a crucial role in global climate regulation through its massive carbon storage capacity.",
    
    "Machine learning algorithms trained on vast datasets of human language can now generate coherent text, translate between languages, answer complex questions, and even write computer code, but they fundamentally operate through statistical pattern recognition rather than true understanding, raising philosophical questions about the nature of intelligence and consciousness.",
    
    "The Great Pacific Garbage Patch, a massive accumulation of plastic debris floating in the North Pacific Ocean between Hawaii and California, covers an area larger than twice the size of Texas and contains an estimated 80,000 metric tons of plastic waste that poses severe threats to marine life through ingestion, entanglement, and toxic chemical leaching.",
    
    "CRISPR-Cas9 gene editing technology allows scientists to make precise modifications to DNA sequences by using a guide RNA to direct the Cas9 enzyme to specific genomic locations where it cuts the double helix, enabling researchers to delete, insert, or replace genetic material with unprecedented accuracy and potentially cure inherited diseases.",
    
    "The ancient Library of Alexandria, established in the 3rd century BCE during the Ptolemaic dynasty, represented humanity's first attempt at universal knowledge collection, housing an estimated 400,000 to 700,000 scrolls on subjects ranging from mathematics and astronomy to medicine and literature, before its gradual decline and eventual destruction over several centuries.",
    
    "Photosynthesis, the fundamental biological process that converts carbon dioxide and water into glucose using sunlight energy captured by chlorophyll molecules, not only provides the chemical energy that powers most life on Earth but also produces the oxygen that makes aerobic respiration possible for complex multicellular organisms.",
    
    "The Manhattan Project, a top-secret research and development program during World War II, brought together over 130,000 workers and scientists across multiple sites in the United States to develop the world's first nuclear weapons, culminating in the atomic bombs dropped on Hiroshima and Nagasaki and fundamentally altering international relations and military strategy.",
    
    "Social media algorithms designed to maximize user engagement often create echo chambers by preferentially showing content that confirms existing beliefs and biases, leading to increased political polarization, reduced exposure to diverse viewpoints, and the rapid spread of misinformation that can have serious consequences for democratic discourse and public health.",
    
    "The human microbiome, consisting of trillions of bacteria, viruses, fungi, and other microorganisms living in and on our bodies, plays essential roles in digestion, immune system function, mental health, and disease resistance, with recent research suggesting that disruptions to this microbial ecosystem may contribute to conditions ranging from obesity to depression.",
    
    "Renewable energy technologies including solar photovoltaic panels, wind turbines, and hydroelectric generators have experienced dramatic cost reductions over the past decade, making clean electricity cheaper than fossil fuels in many markets and offering a viable pathway to reduce greenhouse gas emissions while meeting growing global energy demands.",
    
    "The discovery of DNA's double helix structure by Watson, Crick, Franklin, and Wilkins in 1953 revolutionized biology by revealing how genetic information is stored, replicated, and transmitted across generations, laying the foundation for molecular biology, biotechnology, and our modern understanding of heredity, evolution, and genetic diseases.",
    
    "Urban heat islands, caused by the replacement of natural vegetation with concrete, asphalt, and buildings that absorb and retain solar energy, can make cities 2-5 degrees Celsius warmer than surrounding rural areas, exacerbating air conditioning energy use, air pollution, and heat-related health problems, particularly affecting low-income communities.",
    
    "The development of antibiotics, beginning with Alexander Fleming's accidental discovery of penicillin in 1928, has saved millions of lives by enabling doctors to treat previously fatal bacterial infections, but the overuse and misuse of these drugs has led to the evolution of antibiotic-resistant bacteria that pose an increasingly serious threat to global public health.",
    
    "Artificial neural networks inspired by the structure and function of biological neurons use layers of interconnected nodes to process information, with deep learning architectures containing millions or billions of parameters that can learn complex patterns from data through backpropagation algorithms, enabling breakthroughs in computer vision, natural language processing, and game playing."
]

def get_dataset():
    """Return the complete dataset of sentences."""
    return COMPRESSION_DATASET

def get_random_sentence(seed=None):
    """Get a random sentence from the dataset."""
    import random
    if seed:
        random.seed(seed)
    return random.choice(COMPRESSION_DATASET)

def get_sentence_by_index(index):
    """Get a specific sentence by index."""
    if 0 <= index < len(COMPRESSION_DATASET):
        return COMPRESSION_DATASET[index]
    else:
        raise IndexError(f"Index {index} out of range. Dataset has {len(COMPRESSION_DATASET)} sentences.")

def print_dataset_summary():
    """Print a summary of the dataset."""
    print(f"Dataset contains {len(COMPRESSION_DATASET)} sentences")
    lengths = [len(sentence) for sentence in COMPRESSION_DATASET]
    print(f"Average length: {sum(lengths)/len(lengths):.1f} characters")
    print(f"Shortest: {min(lengths)} characters")
    print(f"Longest: {max(lengths)} characters")
    
    print("\nFirst few sentences:")
    for i, sentence in enumerate(COMPRESSION_DATASET[:3]):
        print(f"{i+1}. {sentence[:100]}...")

if __name__ == "__main__":
    print_dataset_summary() 