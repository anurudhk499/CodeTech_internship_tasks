import spacy
import pytextrank

def initialize_nlp_pipeline():
    """Initialize the NLP pipeline with spaCy and TextRank."""
    # Load the large English language model
    nlp = spacy.load("en_core_web_lg")
    
    # Add TextRank to the pipeline for keyword extraction and summarization
    nlp.add_pipe("textrank")
    
    return nlp

def summarize_text(text, nlp, summary_sentences=2, key_phrases=2):
    """
    Generate a summary of the input text using TextRank algorithm.
    
    Args:
        text (str): The text to summarize
        nlp: The initialized spaCy pipeline
        summary_sentences (int): Number of sentences in the summary
        key_phrases (int): Number of key phrases to consider
        
    Returns:
        dict: Contains the summary, original length, and summary length
    """
    # Process the text with spaCy
    doc = nlp(text)
    
    # Generate the summary
    summary = [str(sent) for sent in doc._.textrank.summary(
        limit_phrases=key_phrases, 
        limit_sentences=summary_sentences
    )]
    
    return {
        "original_length": len(text),
        "summary": "\n".join(summary),
        "summary_length": sum(len(sent) for sent in summary)
    }

def main():
    """Main function to demonstrate text summarization."""
    # Example text about deep learning
    example_text = """Deep learning (also known as deep structured learning) is part of a 
    broader family of machine learning methods based on artificial neural networks with 
    representation learning. Learning can be supervised, semi-supervised or unsupervised. 
    Deep-learning architectures such as deep neural networks, deep belief networks, 
    deep reinforcement learning,recurrent neural networks and convolutional neural networks
    have been applied to fields including computer vision, speech recognition, natural language 
    processing, machine translation, bioinformatics, drug design, medical image analysis, material
    inspection and board game programs, where they have produced results comparable to 
    and in some cases surpassing human expert performance. Artificial neural networks
    (ANNs) were inspired by information processing and distributed communication nodes
    in biological systems. ANNs have various differences from biological brains. Specifically, 
    neural networks tend to be static and symbolic, while the biological brain of most living organisms
    is dynamic (plastic) and analogue. The adjective "deep" in deep learning refers to the use of multiple
    layers in the network. Early work showed that a linear perceptron cannot be a universal classifier, 
    but that a network with a nonpolynomial activation function with one hidden layer of unbounded width can.
    Deep learning is a modern variation which is concerned with an unbounded number of layers of bounded size, 
    which permits practical application and optimized implementation, while retaining theoretical universality 
    under mild conditions. In deep learning the layers are also permitted to be heterogeneous and to deviate widely 
    from biologically informed connectionist models, for the sake of efficiency, trainability and understandability, 
    whence the structured part."""
    
    # Initialize the NLP pipeline
    nlp = initialize_nlp_pipeline()
    
    # Generate and display the summary
    result = summarize_text(example_text, nlp)
    
    print("\n=== ORIGINAL TEXT ===")
    print(f"Character count: {result['original_length']}\n")
    
    print("\n=== SUMMARY ===")
    print(result['summary'])
    print(f"\nSummary character count: {result['summary_length']}")

if __name__ == "__main__":
    main()
