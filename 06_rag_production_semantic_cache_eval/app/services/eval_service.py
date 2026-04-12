import logging
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset

logger = logging.getLogger(__name__)

class RagasEvaluator:
    """
    Evaluates RAG pipeline outputs using RAGAS metrics.
    Note: In production, you would configure RAGAS to use specific LLMs.
    """
    @staticmethod
    def evaluate_response(question: str, answer: str, contexts: list[str]) -> dict:
        logger.info("Starting RAGAS Evaluation...")
        
        # RAGAS requires a HuggingFace Dataset format
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts]
        }
        dataset = Dataset.from_dict(data)
        
        # We evaluate on two core metrics:
        # 1. Faithfulness: Does the answer contradict the context? (Hallucination check)
        # 2. Answer Relevancy: Does the answer actually address the user's question?
        try:
            # We use a try block because RAGAS requires valid API keys configured in the environment 
            # for its internal evaluator LLMs (usually OpenAI by default, can be configured for others)
            result = evaluate(
                dataset,
                metrics=[faithfulness, answer_relevancy],
            )
            
            return {
                "faithfulness_score": result.get("faithfulness", 0.0),
                "answer_relevancy_score": result.get("answer_relevancy", 0.0)
            }
        except Exception as e:
            logger.error(f"RAGAS Evaluation failed (ensure API keys for evaluator LLMs are set): {e}")
            # Returning mock scores if the user doesn't have the heavy RAGAS environment fully configured
            return {
                "faithfulness_score": -1.0,
                "answer_relevancy_score": -1.0
            }