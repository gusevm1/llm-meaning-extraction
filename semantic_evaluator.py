"""
Semantic evaluation module for measuring similarity between original and decompressed sentences.
Uses multiple metrics to assess semantic preservation.
"""

import os
import numpy as np
from typing import Dict, List, Tuple, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import re

# Load environment variables
load_dotenv()

class SemanticEvaluator:
    """Evaluates semantic similarity between original and decompressed sentences."""
    
    def __init__(self, api_key: str = None):
        """Initialize the semantic evaluator with various similarity models."""
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        
        # Initialize sentence transformer for embeddings
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Sentence transformer model loaded")
        except Exception as e:
            print(f"⚠️  Warning: Could not load sentence transformer: {e}")
            self.sentence_model = None
        
        # Initialize OpenAI for LLM-based evaluation
        if self.api_key:
            self.openai_client = OpenAI(api_key=self.api_key)
        else:
            self.openai_client = None
            print("⚠️  Warning: No OpenAI API key found for LLM evaluation")
        
        # Download NLTK data if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("📥 Downloading NLTK data...")
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            print("📥 Downloading NLTK punkt_tab data...")
            nltk.download('punkt_tab', quiet=True)
    
    def embedding_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between sentence embeddings."""
        if not self.sentence_model:
            return 0.0
        
        try:
            embeddings = self.sentence_model.encode([text1, text2])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
        except Exception as e:
            print(f"Error in embedding similarity: {e}")
            return 0.0
    
    def bleu_similarity(self, original: str, decompressed: str) -> float:
        """Calculate BLEU score between original and decompressed text."""
        try:
            # Tokenize sentences
            original_tokens = nltk.word_tokenize(original.lower())
            decompressed_tokens = nltk.word_tokenize(decompressed.lower())
            
            # Calculate BLEU with smoothing
            smoothing = SmoothingFunction()
            bleu_score = sentence_bleu(
                [original_tokens], 
                decompressed_tokens,
                smoothing_function=smoothing.method1
            )
            return float(bleu_score)
        except Exception as e:
            print(f"Error in BLEU calculation: {e}")
            return 0.0
    
    def llm_semantic_similarity(self, original: str, decompressed: str) -> Tuple[float, str]:
        """Use LLM to evaluate semantic similarity and provide reasoning."""
        if not self.openai_client:
            return 0.0, "No OpenAI client available"
        
        prompt = f"""Evaluate the semantic similarity between these two sentences on a scale of 0.0 to 1.0, where:
- 1.0 = Identical meaning, all key information preserved
- 0.8-0.9 = Very similar meaning, minor details might differ
- 0.6-0.7 = Generally similar meaning, some information lost or changed
- 0.4-0.5 = Partially similar, significant information differences
- 0.0-0.3 = Different meanings, major information loss

ORIGINAL: {original}

DECOMPRESSED: {decompressed}

Provide:
1. A similarity score (0.0 to 1.0)
2. Brief explanation of what information was preserved/lost

Format your response as:
SCORE: [number]
ANALYSIS: [explanation]"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",  # Use gpt-4o for semantic evaluation (more cost-effective than o1 for this task)
                messages=[
                    {"role": "system", "content": "You are an expert in semantic analysis. Be precise and objective in your evaluation."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.1
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Extract score and analysis
            score_match = re.search(r'SCORE:\s*([0-9]*\.?[0-9]+)', response_text)
            analysis_match = re.search(r'ANALYSIS:\s*(.+)', response_text, re.DOTALL)
            
            score = float(score_match.group(1)) if score_match else 0.0
            analysis = analysis_match.group(1).strip() if analysis_match else "No analysis provided"
            
            return score, analysis
            
        except Exception as e:
            return 0.0, f"Error in LLM evaluation: {e}"
    
    def keyword_preservation(self, original: str, decompressed: str) -> float:
        """Calculate what percentage of important keywords were preserved."""
        try:
            # Extract important words (longer than 3 characters, not common stop words)
            stop_words = {'the', 'and', 'but', 'for', 'are', 'with', 'that', 'this', 'have', 'from', 'they', 'been', 'their', 'were', 'said', 'each', 'which', 'will', 'would', 'there', 'could', 'when', 'what', 'your', 'more', 'than', 'into', 'some', 'time', 'very', 'only', 'just', 'like', 'over', 'also', 'back', 'after', 'first', 'well', 'year', 'work', 'such', 'make', 'even', 'most', 'take', 'them', 'see', 'him', 'two', 'how', 'its', 'who', 'oil', 'sit', 'now', 'find', 'may', 'say', 'she', 'use', 'her', 'all', 'any', 'can', 'had', 'way', 'day', 'get', 'has', 'old', 'you', 'man', 'new', 'now', 'too', 'many', 'other', 'years', 'great', 'water', 'good', 'know', 'where', 'much', 'come', 'does', 'look', 'think', 'people', 'before', 'both', 'place', 'right', 'through', 'still', 'while', 'being', 'these', 'should', 'again', 'between', 'during', 'under', 'same', 'never', 'every', 'without', 'might', 'another', 'around', 'little', 'long', 'about', 'down', 'world'}
            
            def extract_keywords(text):
                words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
                return set(word for word in words if len(word) > 3 and word not in stop_words)
            
            original_keywords = extract_keywords(original)
            decompressed_keywords = extract_keywords(decompressed)
            
            if not original_keywords:
                return 1.0
            
            preserved_keywords = original_keywords.intersection(decompressed_keywords)
            preservation_ratio = len(preserved_keywords) / len(original_keywords)
            
            return float(preservation_ratio)
            
        except Exception as e:
            print(f"Error in keyword preservation: {e}")
            return 0.0
    
    def evaluate_all(self, original: str, decompressed: str) -> Dict[str, Any]:
        """Run all evaluation metrics and return comprehensive results."""
        print("🔍 Running semantic evaluation...")
        
        results = {}
        
        # 1. Embedding similarity
        print("  📊 Computing embedding similarity...")
        results['embedding_similarity'] = self.embedding_similarity(original, decompressed)
        
        # 2. BLEU score
        print("  📊 Computing BLEU score...")
        results['bleu_score'] = self.bleu_similarity(original, decompressed)
        
        # 3. Keyword preservation
        print("  📊 Computing keyword preservation...")
        results['keyword_preservation'] = self.keyword_preservation(original, decompressed)
        
        # 4. LLM semantic evaluation
        print("  📊 Getting LLM semantic evaluation...")
        llm_score, llm_analysis = self.llm_semantic_similarity(original, decompressed)
        results['llm_similarity'] = llm_score
        results['llm_analysis'] = llm_analysis
        
        # 5. Composite score (weighted average)
        weights = {
            'embedding_similarity': 0.3,
            'bleu_score': 0.2,
            'keyword_preservation': 0.2,
            'llm_similarity': 0.3
        }
        
        composite_score = sum(
            results[metric] * weight 
            for metric, weight in weights.items() 
            if metric in results and results[metric] is not None
        )
        
        results['composite_score'] = composite_score
        results['weights_used'] = weights
        
        return results
    
    def print_evaluation_report(self, results: Dict[str, Any]):
        """Print a formatted evaluation report."""
        print("\n" + "="*60)
        print("🎯 SEMANTIC EVALUATION REPORT")
        print("="*60)
        
        print(f"\n📈 SIMILARITY METRICS:")
        print(f"   Embedding Similarity: {results.get('embedding_similarity', 0):.3f}")
        print(f"   BLEU Score:          {results.get('bleu_score', 0):.3f}")
        print(f"   Keyword Preservation: {results.get('keyword_preservation', 0):.3f}")
        print(f"   LLM Similarity:      {results.get('llm_similarity', 0):.3f}")
        print(f"   📊 COMPOSITE SCORE:   {results.get('composite_score', 0):.3f}")
        
        # Performance interpretation
        composite = results.get('composite_score', 0)
        if composite >= 0.8:
            interpretation = "🟢 Excellent semantic preservation"
        elif composite >= 0.6:
            interpretation = "🟡 Good semantic preservation"
        elif composite >= 0.4:
            interpretation = "🟠 Moderate semantic preservation"
        else:
            interpretation = "🔴 Poor semantic preservation"
        
        print(f"\n🎭 INTERPRETATION: {interpretation}")
        
        # Show LLM analysis if available
        if 'llm_analysis' in results and results['llm_analysis']:
            print(f"\n🤖 LLM ANALYSIS:")
            print(f"   {results['llm_analysis']}")

def test_evaluator():
    """Test the semantic evaluator with sample sentences."""
    evaluator = SemanticEvaluator()
    
    original = "The quick brown fox jumps over the lazy dog in the beautiful meadow."
    decompressed = "A fast brown fox leaps over a sleepy dog in a lovely field."
    
    results = evaluator.evaluate_all(original, decompressed)
    evaluator.print_evaluation_report(results)

if __name__ == "__main__":
    test_evaluator() 