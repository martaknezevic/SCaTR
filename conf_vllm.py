import openai
import numpy as np
import asyncio

async def generate_single_response(client, config):
    """Generate a single response asynchronously"""
    response = await client.chat.completions.create(**config)
    return response.choices[0]  # Return just the choice since we're not using n anymore

async def generate_multiple_responses(client, config, n_gen):
    """Generate multiple responses concurrently"""
    # Create tasks for concurrent generation
    tasks = [generate_single_response(client, config) for _ in range(n_gen)]
    
    # Wait for all tasks to complete
    choices = await asyncio.gather(*tasks)
    
    return choices

def calculate_tail_confidence(choice, tail_n=512):
    """
    Calculate tail confidence for a single trace.
    
    Args:
        choice: OpenAI choice object with logprobs
        tail_n: Number of bottom tokens to consider for tail confidence
    
    Returns:
        dict with confidence metrics
    """
    if not choice.logprobs or not choice.logprobs.content:
        return {
            'response': choice.message.content,
            'tail_confidence': 0.0,
            'mean_confidence': 0.0,
            'tokens': [],
            'confidences': []
        }
    
    tokens = []
    confidences = []
    
    # Convert logprobs to confidence
    for token_data in choice.logprobs.content:
        tokens.append(token_data.token)
        
        if token_data.top_logprobs:
            # Mean confidence = negative average of top logprobs
            mean_conf = -sum(logprob_data.logprob for logprob_data in token_data.top_logprobs) / len(token_data.top_logprobs)
            confidences.append(mean_conf)
        else:
            # Fallback if no top_logprobs available
            confidences.append(-token_data.logprob)
    
    # Calculate tail confidence (average of bottom N tokens)
    if len(confidences) >= tail_n:
        tail_confidence = np.mean(confidences[-tail_n:])
    else:
        tail_confidence = np.mean(confidences) if confidences else 0.0
    
    return {
        'response': choice.message.content,
        'tail_confidence': tail_confidence,
        'mean_confidence': np.mean(confidences) if confidences else 0.0,
        'tokens': tokens,
        'confidences': confidences,
        'total_tokens': len(tokens)
    }

def select_best_by_tail_confidence(choices, tail_n=512):
    """
    Select the best response based on tail confidence.
    
    Args:
        choices: List of OpenAI choice objects
        tail_n: Number of bottom tokens to consider
    
    Returns:
        tuple: (best_response_text, all_results, best_index)
    """
    results = []
    
    # Calculate tail confidence for each choice
    for i, choice in enumerate(choices):
        result = calculate_tail_confidence(choice, tail_n)
        result['choice_index'] = i
        results.append(result)
    
    # Sort by tail confidence (highest first)
    results.sort(key=lambda x: x['tail_confidence'], reverse=True)
    best_result = results[0]
    
    return best_result['response'], results, best_result['choice_index']

async def main():
    # Main Entry
    prompt = """
How many r's are there in the word 'strawberry'?
"""

    client = openai.AsyncOpenAI(
        base_url="http://localhost:9000/v1", #change port number
        api_key="token-abc123",
        timeout=30 * 60,
    )

    # CONSTANTS
    models = await client.models.list()
    model_id = models.data[0].id
    N_GEN = 5 # Total parallel generations
    TAIL_N = 2048  # Number of tokens to consider for tail confidence

    config = {
        "model": model_id,
        "max_tokens": 8192,
        "temperature": 1.0,
        "messages": [{"role": "user", "content": prompt}],
        "logprobs": True,               
        "top_logprobs": 10,
    }

    # Generate responses asynchronously
    print(f"Generating {N_GEN} responses asynchronously...")
    choices = await generate_multiple_responses(client, config, N_GEN)
    


    # Select best response based on tail confidence
    best_response, all_results, best_index = select_best_by_tail_confidence(choices, tail_n=TAIL_N)

    # Show all results sorted by tail confidence
    for i, result in enumerate(all_results):
        print("*"*100)
        print(f"Rank {i+1} (Original choice {result['choice_index']}):")
        print(f"  Response: {result['response']}")
        print(f"  Tail confidence: {result['tail_confidence']:.4f}")
        print(f"  Mean confidence: {result['mean_confidence']:.4f}")
        print(f"  Total tokens: {result['total_tokens']}")
        print("*"*100)
        print()

    print("=" * 150)
    print(f"BEST RESPONSE (Highest tail-{TAIL_N} confidence):")
    print(f"Choice {best_index} with tail confidence: {all_results[0]['tail_confidence']:.4f}")
    print(f"Response: {best_response}")
    print("=" * 150)

if __name__ == "__main__":
    asyncio.run(main())