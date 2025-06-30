"""
Extreme compression experiment with non-human readable symbols using Google Gemini models.
Tests compression and blind decompression with separate model instances.
"""

import os
import sys
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from dataset import get_sentence_by_index, get_dataset
from semantic_evaluator import SemanticEvaluator
import json
import time
import csv
import pandas as pd
import numpy as np

# Load environment variables
load_dotenv()

class ExtremeCompressionExperimentGemini:
    """Handler for extreme compression experiments with blind decompression using Gemini models."""
    
    def __init__(self, min_compression_ratio: float = 0.15):
        """Initialize with Gemini client using direct API approach."""
        
        # Get API key from environment
        self.api_key = os.environ.get('GOOGLE_API_KEY') or os.environ.get('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("No API key found! Set GOOGLE_API_KEY in your .env file")
        
        # Initialize Google Gen AI client (direct API - the working approach)
        try:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
            print(f"✅ Gemini client initialized with API key: {self.api_key[:15]}...")
        except ImportError:
            raise ImportError("Google Gen AI SDK not installed. Run: pip install google-genai")
        
        # Test connection with Gemini 2.5 Flash Lite
        self.model = "gemini-2.5-pro"  # Using the Gemini 2.5 Pro model!
        if not self._test_connection():
            raise ConnectionError("Could not establish connection to Gemini 2.5 Flash Lite")
        
        self.min_compression_ratio = min_compression_ratio  # Must compress to 10% or less
        
        # Initialize semantic evaluator (using OpenAI for evaluation consistency)
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            print("⚠️  Warning: OpenAI API key not found. Semantic evaluation may be limited.")
        self.semantic_evaluator = SemanticEvaluator(openai_api_key) if openai_api_key else None
    
    def _test_connection(self) -> bool:
        """Test connection to Gemini API."""
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents="Say 'Connected' if you can hear me"
            )
            if response.text and "Connected" in response.text:
                print(f"✅ {self.model} connection successful!")
                return True
            else:
                print(f"❌ {self.model} connection failed: No valid response")
                return False
        except Exception as e:
            print(f"❌ {self.model} connection failed: {e}")
            return False
    
    def _generate_text(self, prompt: str, max_tokens: int = 300, temperature: float = 0.1) -> str:
        """Generate text using Gemini API with the working direct approach."""
        try:
            # Use the same simple format that works in test_direct_api.py
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            
            if response.text:
                return response.text.strip()
            else:
                print(f"Warning: Empty response from {self.model}")
                return ""
                
        except Exception as e:
            print(f"Error generating text with {self.model}: {e}")
            return ""

    def extreme_compress(self, sentence: str, max_attempts: int = 7) -> Dict[str, Any]:
        """
        Perform extreme compression with enforced minimum compression ratio.
        
        Args:
            sentence: The sentence to compress
            max_attempts: Maximum number of compression attempts
        
        Returns:
            Dictionary with compression results and metadata
        """
        original_length = len(sentence)
        target_length = int(original_length * self.min_compression_ratio)
        
        compression_results = {
            "attempts": [],
            "final_compressed": None,
            "achieved_ratio": None,
            "target_ratio": self.min_compression_ratio,
            "success": False
        }
        
        for attempt in range(1, max_attempts + 1):
            print(f"   Attempt {attempt}/{max_attempts} (target: ≤{target_length} chars)")
            
            # Create system context and user prompt - matching OpenAI approach exactly
            if attempt == 1:
                system_context = "You are a text compression expert. Focus on preserving complete meaning while reducing length. Unicode characters are available. Output ONLY the compressed text, no explanations."
                user_prompt = f"""Please compress this sentence while preserving all the important information. Aim for around {target_length} characters. You can use any characters including unicode symbols:

{sentence}"""
            elif attempt == 2:
                prev_compressed = compression_results["attempts"][-1]["compressed"] if compression_results["attempts"] else sentence
                prev_length = compression_results["attempts"][-1]["length"] if compression_results["attempts"] else len(sentence)
                system_context = f"You are a compression specialist. The output must be {target_length} characters or fewer. Unicode symbols can encode dense information. Count characters carefully. Output ONLY compressed text, no explanations."
                user_prompt = f"""The previous compression was {prev_length} characters, but I need it to be {target_length} characters or less. Consider using unicode characters that can represent concepts more densely:

Previous attempt: {prev_compressed}

Make it much more compact while preserving ALL the semantic content."""
            elif attempt == 3:
                prev_compressed = compression_results["attempts"][-1]["compressed"] if compression_results["attempts"] else sentence
                prev_length = compression_results["attempts"][-1]["length"] if compression_results["attempts"] else len(sentence)
                system_context = f"You are a compression specialist. The output must be {target_length} characters or fewer. Unicode symbols can encode dense information. Count characters carefully. Output ONLY compressed text, no explanations."
                user_prompt = f"""Still too long at {prev_length} characters. I absolutely need {target_length} characters or fewer. Think creatively about unicode symbols, mathematical notation, or other dense character representations:

{prev_compressed}

Target: {target_length} characters maximum. Every character must carry maximum meaning."""
            elif attempt <= 5:
                prev_compressed = compression_results["attempts"][-1]["compressed"] if compression_results["attempts"] else sentence
                prev_length = compression_results["attempts"][-1]["length"] if compression_results["attempts"] else len(sentence)
                system_context = f"CRITICAL: Output must be exactly {target_length} characters or less. Use unicode as semantic containers - each character should carry maximum meaning density. Output ONLY compressed text, no explanations."
                user_prompt = f"""CRITICAL: This must be {target_length} characters or less. Consider unicode characters that can encode multiple concepts - think of them as dense information containers rather than just symbols:

{prev_compressed}

Use unicode creatively to preserve the FULL semantic meaning in just {target_length} characters."""
            else:
                prev_compressed = compression_results["attempts"][-1]["compressed"] if compression_results["attempts"] else sentence
                prev_length = compression_results["attempts"][-1]["length"] if compression_results["attempts"] else len(sentence)
                system_context = f"CRITICAL: Output must be exactly {target_length} characters or less. Use unicode as semantic containers - each character should carry maximum meaning density. Output ONLY compressed text, no explanations."
                user_prompt = f"""FINAL ATTEMPT: {target_length} characters maximum. Use unicode characters as semantic vessels - each character should encode as much meaning as possible. The goal is to preserve the complete meaning of the original sentence, not just keywords:

{prev_compressed}

Create a {target_length}-character representation that captures the ENTIRE semantic essence."""
            
            # Combine system context and user prompt for Gemini (since it doesn't have separate system messages)
            prompt = f"{system_context}\n\n{user_prompt}"

            try:
                # Gemini 2.5 Pro generation parameters
                max_tokens = min(target_length + 50, 300)
                temperature = 0.1 + (attempt * 0.1)  # Gradually increase creativity
                
                compressed = self._generate_text(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                if not compressed:
                    print(f"     Error: Empty response from Gemini 2.5 Flash Lite")
                    continue
                
                compressed_length = len(compressed)
                ratio = compressed_length / original_length
                
                compression_results["attempts"].append({
                    "attempt": attempt,
                    "compressed": compressed,
                    "length": compressed_length,
                    "ratio": ratio,
                    "meets_target": compressed_length <= target_length
                })
                
                print(f"     Result: {compressed_length} chars (ratio: {ratio:.3f})")
                
                if compressed_length <= target_length:
                    print(f"     ✅ Target achieved!")
                    compression_results["final_compressed"] = compressed
                    compression_results["achieved_ratio"] = ratio
                    compression_results["success"] = True
                    break
                else:
                    print(f"     ❌ Still too long ({compressed_length} > {target_length})")
                    
            except Exception as e:
                print(f"     Error in attempt {attempt}: {e}")
                continue
        
        # If no attempt succeeded, use the best attempt
        if not compression_results["success"] and compression_results["attempts"]:
            best_attempt = min(compression_results["attempts"], key=lambda x: x["length"])
            compression_results["final_compressed"] = best_attempt["compressed"]
            compression_results["achieved_ratio"] = best_attempt["ratio"]
            print(f"   ⚠️  Using best attempt: {best_attempt['length']} chars (ratio: {best_attempt['ratio']:.3f})")
        
        return compression_results

    def blind_decompress(self, compressed_text: str) -> str:
        """
        Perform blind decompression without seeing the original sentence.
        
        Args:
            compressed_text: The compressed representation to expand
        
        Returns:
            Decompressed natural language sentence
        """
        # Match OpenAI's system/user message structure  
        system_context = "You are a decompression expert. Output ONLY the reconstructed sentence in natural English, no explanations."
        user_prompt = """You are an expert decompression algorithm. You will receive an extremely compressed representation that may contain unicode characters, symbols, abbreviations, or dense encodings where each character potentially carries significant semantic meaning.

YOUR TASK:
- Analyze each character carefully - unicode symbols may encode entire concepts
- Look for patterns where single characters represent complex ideas
- Reconstruct the complete original meaning, not just keywords
- The compression preserved FULL semantic content in very few characters
- Expand everything back to complete, natural English
- Capture all nuances and relationships from the original

The compressed input uses maximum information density - every character matters. Your job is to unfold the complete semantic meaning.

Compressed input to decompress:"""
        
        # Combine for Gemini
        prompt = f"{system_context}\n\n{user_prompt}\n\n{compressed_text}"

        try:
            decompressed = self._generate_text(
                prompt=prompt,
                max_tokens=400,
                temperature=0.1  # Low temperature for consistency
            )
            return decompressed if decompressed else compressed_text
        except Exception as e:
            print(f"Decompression failed: {e}")
            return compressed_text

    def calculate_metrics(self, original: str, compressed: str, decompressed: str) -> Dict[str, Any]:
        """Calculate compression and quality metrics."""
        return {
            "original_length": len(original),
            "compressed_length": len(compressed),
            "decompressed_length": len(decompressed),
            "compression_ratio": len(compressed) / len(original) if len(original) > 0 else 0,
            "expansion_ratio": len(decompressed) / len(compressed) if len(compressed) > 0 else 0,
            "recovery_ratio": len(decompressed) / len(original) if len(original) > 0 else 0
        }

    def run_single_experiment(self, sentence_index: int = 0) -> Dict[str, Any]:
        """
        Run a complete experiment on a single sentence.
        
        Args:
            sentence_index: Index of sentence from dataset to test
        
        Returns:
            Complete experiment results
        """
        original_sentence = get_sentence_by_index(sentence_index)
        
        print(f"🧪 Running Extreme Compression Experiment (Gemini)")
        print(f"📝 Sentence {sentence_index + 1} from dataset")
        print(f"🎯 Target compression: ≤{self.min_compression_ratio*100:.0f}% of original size")
        print(f"🤖 Using model: {self.model}")
        print("=" * 80)
        
        # Step 1: Extreme Compression with enforced ratio
        print("🔄 Step 1: Extreme Compression...")
        compression_results = self.extreme_compress(original_sentence)
        
        if not compression_results["final_compressed"]:
            print("❌ Compression failed completely!")
            return {"error": "Compression failed", "sentence_index": sentence_index}
        
        compressed = compression_results["final_compressed"]
        
        # Small delay to simulate separate instances
        time.sleep(1)
        
        # Step 2: Blind Decompression
        print("🔄 Step 2: Blind Decompression...")
        decompressed = self.blind_decompress(compressed)
        
        # Step 3: Semantic Evaluation (if available)
        semantic_results = {}
        if self.semantic_evaluator:
            print("🔄 Step 3: Semantic Evaluation...")
            semantic_results = self.semantic_evaluator.evaluate_all(original_sentence, decompressed)
        else:
            print("⚠️  Step 3: Semantic evaluation skipped (OpenAI API key not available)")
            semantic_results = {
                "embedding_similarity": 0,
                "bleu_score": 0,
                "keyword_preservation": 0,
                "llm_similarity": 0,
                "composite_score": 0,
                "llm_analysis": "Evaluation skipped - OpenAI API key not available"
            }
        
        # Calculate basic metrics
        basic_metrics = self.calculate_metrics(original_sentence, compressed, decompressed)
        
        # Prepare comprehensive results
        results = {
            "sentence_index": sentence_index,
            "original": original_sentence,
            "compressed": compressed,
            "decompressed": decompressed,
            "compression_details": compression_results,
            "basic_metrics": basic_metrics,
            "semantic_evaluation": semantic_results,
            "timestamp": time.time(),
            "min_compression_target": self.min_compression_ratio,
            "model_used": self.model,
            "provider": "gemini"
        }
        
        # Display results
        self.display_results(results)
        
        return results

    def display_results(self, results: Dict[str, Any]):
        """Display experiment results in a formatted way."""
        print("\n" + "="*80)
        print("📊 EXPERIMENT RESULTS (Gemini)")
        print("="*80)
        
        # Check if this is an error result
        if "error" in results:
            print(f"❌ {results['error']}")
            return
        
        print(f"\n📝 ORIGINAL ({results['basic_metrics']['original_length']} chars):")
        print(f"   {results['original']}")
        
        print(f"\n🗜️  COMPRESSED ({results['basic_metrics']['compressed_length']} chars):")
        print(f"   {results['compressed']}")
        
        print(f"\n📤 DECOMPRESSED ({results['basic_metrics']['decompressed_length']} chars):")
        print(f"   {results['decompressed']}")
        
        # Compression Performance
        print(f"\n📈 COMPRESSION PERFORMANCE:")
        basic_metrics = results['basic_metrics']
        compression_details = results['compression_details']
        
        print(f"   Model Used: {results.get('model_used', 'unknown')} (Gemini)")
        print(f"   Target Ratio: ≤{results['min_compression_target']*100:.0f}%")
        print(f"   Achieved Ratio: {basic_metrics['compression_ratio']:.3f} ({basic_metrics['compression_ratio']*100:.1f}% of original)")
        print(f"   Compression Success: {'✅ YES' if compression_details['success'] else '❌ NO'}")
        print(f"   Attempts Used: {len(compression_details['attempts'])}")
        
        char_savings = basic_metrics['original_length'] - basic_metrics['compressed_length']
        print(f"   Characters Saved: {char_savings} ({char_savings/basic_metrics['original_length']*100:.1f}%)")
        
        # Show semantic evaluation results
        if 'semantic_evaluation' in results and self.semantic_evaluator:
            self.semantic_evaluator.print_evaluation_report(results['semantic_evaluation'])
        else:
            print(f"\n🎭 SEMANTIC EVALUATION: Skipped (OpenAI API key required)")

    def flatten_results_for_csv(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Convert nested results to flat structure for CSV export."""
        if "error" in results:
            return {
                "sentence_index": results.get("sentence_index", -1),
                "error": results["error"],
                "timestamp": results.get("timestamp", time.time()),
                "provider": "gemini"
            }
        
        basic_metrics = results.get("basic_metrics", {})
        semantic_eval = results.get("semantic_evaluation", {})
        compression_details = results.get("compression_details", {})
        
        # Determine semantic interpretation
        composite_score = semantic_eval.get("composite_score", 0)
        if composite_score >= 0.8:
            interpretation = "Excellent"
        elif composite_score >= 0.6:
            interpretation = "Good"
        elif composite_score >= 0.4:
            interpretation = "Moderate"
        else:
            interpretation = "Poor"
        
        return {
            # Basic info
            "sentence_index": results.get("sentence_index", -1),
            "timestamp": results.get("timestamp", time.time()),
            "model_used": results.get("model_used", "unknown"),
            "provider": "gemini",
            
            # Text content
            "original": results.get("original", ""),
            "compressed": results.get("compressed", ""),
            "decompressed": results.get("decompressed", ""),
            
            # Compression metrics
            "target_compression_ratio": results.get("min_compression_target", 0),
            "achieved_compression_ratio": basic_metrics.get("compression_ratio", 0),
            "compression_success": compression_details.get("success", False),
            "compression_attempts": len(compression_details.get("attempts", [])),
            
            # Length metrics
            "original_length": basic_metrics.get("original_length", 0),
            "compressed_length": basic_metrics.get("compressed_length", 0),
            "decompressed_length": basic_metrics.get("decompressed_length", 0),
            "characters_saved": basic_metrics.get("original_length", 0) - basic_metrics.get("compressed_length", 0),
            "compression_percentage": (1 - basic_metrics.get("compression_ratio", 0)) * 100,
            
            # Semantic evaluation scores
            "embedding_similarity": semantic_eval.get("embedding_similarity", 0),
            "bleu_score": semantic_eval.get("bleu_score", 0),
            "keyword_preservation": semantic_eval.get("keyword_preservation", 0),
            "llm_similarity": semantic_eval.get("llm_similarity", 0),
            "composite_score": semantic_eval.get("composite_score", 0),
            "semantic_interpretation": interpretation,
            "llm_analysis": semantic_eval.get("llm_analysis", ""),
            
            # Recovery metrics
            "expansion_ratio": basic_metrics.get("expansion_ratio", 0),
            "recovery_ratio": basic_metrics.get("recovery_ratio", 0)
        }
    
    def save_to_csv(self, results: Dict[str, Any], csv_filename: str = None, append: bool = True):
        """Save results to CSV file."""
        if csv_filename is None:
            csv_filename = "compression_experiments_gemini.csv"
        
        flat_results = self.flatten_results_for_csv(results)
        
        # Check if file exists and has data
        file_exists = os.path.exists(csv_filename)
        
        mode = 'a' if append and file_exists else 'w'
        
        with open(csv_filename, mode, newline='', encoding='utf-8') as csvfile:
            fieldnames = flat_results.keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header only if file is new or we're overwriting
            if not file_exists or not append:
                writer.writeheader()
            
            writer.writerow(flat_results)
        
        if file_exists and append:
            print(f"📊 Results appended to: {csv_filename}")
        else:
            print(f"📊 Results saved to new CSV: {csv_filename}")
    
    def save_results(self, results: Dict[str, Any], base_filename: str = None, csv_filename: str = None):
        """Save results to both JSON and CSV formats."""
        timestamp = int(time.time())
        
        if base_filename is None:
            json_filename = f"experiment_results_gemini_{timestamp}.json"
        else:
            json_filename = f"{base_filename}_gemini.json"
        
        # Use provided CSV filename or create timestamped one
        if csv_filename is None:
            from datetime import datetime
            readable_time = datetime.fromtimestamp(timestamp).strftime("%Y%m%d_%H%M%S")
            csv_filename = f"compression_experiments_gemini_{readable_time}.csv"
        
        # Save JSON (detailed results)
        with open(json_filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Detailed results saved to: {json_filename}")
        
        # Save CSV (for analysis)
        self.save_to_csv(results, csv_filename, append=True)
    
    def run_full_dataset_experiment(self) -> Dict[str, Any]:
        """
        Run compression experiments on all 20 sentences in the dataset.
        
        Returns:
            Summary results and CSV filename
        """
        print(f"🚀 Running Gemini compression experiments on all 20 dataset sentences")
        print(f"🎯 Target: {self.min_compression_ratio*100:.0f}% compression with semantic preservation")
        print("=" * 80)
        
        # Create timestamped CSV filename
        timestamp = int(time.time())
        from datetime import datetime
        readable_time = datetime.fromtimestamp(timestamp).strftime("%Y%m%d_%H%M%S")
        csv_filename = f"compression_experiments_gemini_{readable_time}.csv"
        
        summary_results = {
            "csv_filename": csv_filename,
            "timestamp": timestamp,
            "total_sentences": 20,
            "completed_experiments": 0,
            "successful_compressions": 0,
            "failed_experiments": 0,
            "compression_ratios": [],
            "semantic_scores": [],
            "errors": [],
            "provider": "gemini"
        }
        
        print(f"📊 Results will be saved to: {csv_filename}")
        print()
        
        for sentence_index in range(20):
            try:
                print(f"\n{'='*20} SENTENCE {sentence_index + 1}/20 {'='*20}")
                
                # Run single experiment
                results = self.run_single_experiment(sentence_index)
                
                # Save to CSV immediately
                self.save_to_csv(results, csv_filename, append=True)
                
                # Update summary
                summary_results["completed_experiments"] += 1
                
                if "error" not in results:
                    summary_results["successful_compressions"] += 1
                    summary_results["compression_ratios"].append(results["basic_metrics"]["compression_ratio"])
                    summary_results["semantic_scores"].append(results["semantic_evaluation"]["composite_score"])
                else:
                    summary_results["failed_experiments"] += 1
                    summary_results["errors"].append(f"Sentence {sentence_index}: {results['error']}")
                
                # Print progress
                success_rate = summary_results["successful_compressions"] / summary_results["completed_experiments"] * 100
                print(f"\n📈 Progress: {sentence_index + 1}/20 completed ({success_rate:.1f}% success rate)")
                
                # Small delay between experiments to avoid rate limits
                time.sleep(2)
                
            except Exception as e:
                print(f"❌ Error processing sentence {sentence_index + 1}: {e}")
                summary_results["failed_experiments"] += 1
                summary_results["errors"].append(f"Sentence {sentence_index}: {str(e)}")
                continue
        
        # Print final summary
        self.print_experiment_summary(summary_results)
        
        return summary_results
    
    def print_experiment_summary(self, summary: Dict[str, Any]):
        """Print a summary of the full dataset experiment."""
        print("\n" + "="*80)
        print("🎯 FULL DATASET EXPERIMENT SUMMARY (Gemini)")
        print("="*80)
        
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"   Total Sentences: {summary['total_sentences']}")
        print(f"   Completed Experiments: {summary['completed_experiments']}")
        print(f"   Successful Compressions: {summary['successful_compressions']}")
        print(f"   Failed Experiments: {summary['failed_experiments']}")
        print(f"   Success Rate: {(summary['successful_compressions']/summary['completed_experiments']*100):.1f}%")
        
        if summary["compression_ratios"]:
            ratios = summary["compression_ratios"]
            print(f"\n🗜️  COMPRESSION PERFORMANCE:")
            print(f"   Mean Compression Ratio: {np.mean(ratios):.3f}")
            print(f"   Best Compression: {min(ratios):.3f}")
            print(f"   Worst Compression: {max(ratios):.3f}")
            print(f"   Standard Deviation: {np.std(ratios):.3f}")
        
        if summary["semantic_scores"]:
            scores = summary["semantic_scores"]
            print(f"\n🎭 SEMANTIC PRESERVATION:")
            print(f"   Mean Semantic Score: {np.mean(scores):.3f}")
            print(f"   Best Semantic Score: {max(scores):.3f}")
            print(f"   Worst Semantic Score: {min(scores):.3f}")
            print(f"   Standard Deviation: {np.std(scores):.3f}")
        
        if summary["errors"]:
            print(f"\n❌ ERRORS:")
            for error in summary["errors"][:5]:  # Show first 5 errors
                print(f"   {error}")
            if len(summary["errors"]) > 5:
                print(f"   ... and {len(summary['errors']) - 5} more errors")
        
        print(f"\n📁 Full results saved to: {summary['csv_filename']}")
        print(f"📈 Run analysis with: python analyze_results.py {summary['csv_filename']}")

    def load_csv_results(self, csv_filename: str = "compression_experiments_gemini.csv") -> pd.DataFrame:
        """Load and return CSV results as a pandas DataFrame for analysis."""
        try:
            df = pd.read_csv(csv_filename)
            print(f"📈 Loaded {len(df)} experiments from {csv_filename}")
            return df
        except FileNotFoundError:
            print(f"❌ File {csv_filename} not found")
            return pd.DataFrame()
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            return pd.DataFrame()

def main():
    """Main function to run the extreme compression experiment."""
    print("🚀 EXTREME COMPRESSION EXPERIMENT (GEMINI)")
    print("Testing ultra-dense compression with unicode semantic encoding")
    print("Using Gemini 2.5 Flash for fast and reliable compression capabilities")
    print("Target: 10% compression ratio with full semantic preservation")
    print("="*80)
    
    try:
        # Initialize experiment
        experiment = ExtremeCompressionExperimentGemini()
        
        # Run experiments on all 20 sentences
        summary = experiment.run_full_dataset_experiment()
        
        print(f"\n✅ Full dataset experiment completed!")
        print(f"📊 {summary['successful_compressions']}/{summary['total_sentences']} experiments successful")
        if summary['compression_ratios']:
            print(f"🎯 Average compression: {np.mean(summary['compression_ratios'])*100:.1f}%")
            print(f"🎭 Average semantic score: {np.mean(summary['semantic_scores']):.3f}")
        print(f"📁 Results saved to: {summary['csv_filename']}")
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        print("\nMake sure:")
        print("1. GOOGLE_API_KEY is set in your .env file")
        print("2. Google Gen AI SDK is installed: pip install google-genai")
        print("3. Dependencies are installed: pip install -r requirements.txt")
        print("4. Test API connection first: python test_direct_api.py")

if __name__ == "__main__":
    main() 