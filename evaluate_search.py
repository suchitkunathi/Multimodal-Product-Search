
import sys
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from clip_encoder import CLIPEncoder
from faiss_index import FAISSIndex


# Define test queries with expected category matches
TEST_QUERIES = [
    {
        "query": "blue cotton shirt",
        "expected_categories": ["Clothing"],
        "expected_keywords": ["shirt", "cotton", "blue"],
        "k": 10
    },
    {
        "query": "running shoes black",
        "expected_categories": ["Footwear"],
        "expected_keywords": ["shoes", "running", "black"],
        "k": 10
    },
    {
        "query": "leather wallet brown",
        "expected_categories": ["Accessories"],
        "expected_keywords": ["wallet", "leather", "brown"],
        "k": 10
    },
    {
        "query": "backpack laptop",
        "expected_categories": ["Bags"],
        "expected_keywords": ["backpack", "laptop"],
        "k": 10
    },
    {
        "query": "black polo shirt",
        "expected_categories": ["Clothing"],
        "expected_keywords": ["polo", "shirt", "black"],
        "k": 10
    },
    {
        "query": "wireless earbuds",
        "expected_categories": ["Electronics"],
        "expected_keywords": ["earbuds", "wireless"],
        "k": 10
    },
]


def is_relevant(result, test_query):
    """Check if result is relevant to query"""
    result_text = f"{result['name']} {result.get('category', '')}".lower()
    
    # Check category match
    category_match = any(cat.lower() in result_text for cat in test_query['expected_categories'])
    
    # Check keyword match
    keyword_matches = sum(1 for kw in test_query['expected_keywords'] 
                         if kw.lower() in result_text)
    
    # At least category OR 2 keywords must match
    return category_match or keyword_matches >= 1


def calculate_precision_at_k(results, test_query, k):
    """Calculate precision@k"""
    relevant = sum(1 for r in results[:k] if is_relevant(r, test_query))
    return relevant / k if k > 0 else 0


def calculate_recall_at_k(results, test_query, k, total_relevant):
    """Calculate recall@k"""
    relevant = sum(1 for r in results[:k] if is_relevant(r, test_query))
    return relevant / total_relevant if total_relevant > 0 else 0


def calculate_mrr(results, test_query):
    """Calculate Mean Reciprocal Rank"""
    for i, result in enumerate(results, 1):
        if is_relevant(result, test_query):
            return 1 / i
    return 0


def calculate_ndcg(results, test_query, k):
    """Calculate Normalized Discounted Cumulative Gain"""
    # Ideal DCG (all relevant items at top)
    idcg = sum(1 / np.log2(i + 1) for i in range(1, min(k + 1, len(results) + 1)))
    
    # Actual DCG
    dcg = sum((1 / np.log2(i + 1)) for i, r in enumerate(results[:k], 1) 
              if is_relevant(r, test_query))
    
    return dcg / idcg if idcg > 0 else 0


def evaluate_search():
    """Run evaluation on test queries"""
    
    print("\n" + "="*80)
    print("🔍 SEARCH ACCURACY EVALUATION")
    print("="*80)
    
    # Load encoder and index
    print("\n📦 Loading models...")
    encoder = CLIPEncoder(model_name="ViT-B/32")
    index = FAISSIndex(embedding_dim=512)
    
    index_path = Path("data/index/products")
    if not index_path.with_suffix('.index').exists():
        print("❌ Index not found! Run: python enhanced-build-index.py")
        return
    
    index.load(str(index_path))
    print(f"✓ Loaded index with {index.index.ntotal} products")
    
    # Evaluate each test query
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    results_summary = {
        "precision@10": [],
        "recall@10": [],
        "mrr": [],
        "ndcg@10": [],
    }
    
    for i, test in enumerate(tqdm(TEST_QUERIES), 1):
        query = test["query"]
        k = test["k"]
        
        # Get search results
        query_embedding = encoder.encode_text(query)
        results = index.search(query_embedding, k=k)
        
        # Calculate metrics
        precision = calculate_precision_at_k(results, test, k)
        recall = calculate_recall_at_k(results, test, k, k)  # Approximate
        mrr = calculate_mrr(results, test)
        ndcg = calculate_ndcg(results, test, k)
        
        results_summary["precision@10"].append(precision)
        results_summary["recall@10"].append(recall)
        results_summary["mrr"].append(mrr)
        results_summary["ndcg@10"].append(ndcg)
        
        print(f"\n{'─'*80}")
        print(f"Query {i}: \"{query}\"")
        print(f"{'─'*80}")
        print(f"Expected: {', '.join(test['expected_categories'])}")
        print(f"\nMetrics:")
        print(f"  • Precision@10:  {precision:.1%} (How many results are relevant?)")
        print(f"  • Recall@10:     {recall:.1%} (Did we find the relevant items?)")
        print(f"  • MRR:           {mrr:.3f} (How soon is first relevant item?)")
        print(f"  • NDCG@10:       {ndcg:.3f} (Overall ranking quality 0-1)")
        
        print(f"\nTop 3 Results:")
        for j, result in enumerate(results[:3], 1):
            is_rel = "✓" if is_relevant(result, test) else "✗"
            print(f"  {j}. {is_rel} {result['name']} ({result['category']}) - ${result['price']:.2f}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")
    print(f"\nAverage Metrics across {len(TEST_QUERIES)} queries:")
    print(f"  • Precision@10:  {np.mean(results_summary['precision@10']):.1%}")
    print(f"  • Recall@10:     {np.mean(results_summary['recall@10']):.1%}")
    print(f"  • MRR:           {np.mean(results_summary['mrr']):.3f}")
    print(f"  • NDCG@10:       {np.mean(results_summary['ndcg@10']):.3f}")
    
    print(f"\n{'='*80}")
    print("✅ EVALUATION COMPLETE")
    print(f"{'='*80}\n")
    
    # Save results
    results_file = Path("evaluation_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": str(Path.cwd()),
            "summary": {
                "precision@10": float(np.mean(results_summary['precision@10'])),
                "recall@10": float(np.mean(results_summary['recall@10'])),
                "mrr": float(np.mean(results_summary['mrr'])),
                "ndcg@10": float(np.mean(results_summary['ndcg@10'])),
            },
            "details": results_summary
        }, f, indent=2)
    
    print(f"📊 Results saved to evaluation_results.json\n")


if __name__ == "__main__":
    evaluate_search()