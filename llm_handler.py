import logging
import mlx.core as mx
from mlx_lm import load, generate

logger = logging.getLogger("MLX_Whisper_Processor.LLMHandler") # Child logger

# Error indicators returned by invoke_llm_mlx
INPUT_TOO_LONG_ERROR_INDICATOR = "ERROR:INPUT_TOO_LONG_FOR_MODEL_CONTEXT_WINDOW"
PROMPT_TOKENIZATION_FAILED_INDICATOR = "ERROR:PROMPT_TOKENIZATION_FAILED"
GENERATION_OOM_ERROR_INDICATOR = "ERROR:GENERATION_OUT_OF_MEMORY"
# Other generic failure can still return None

def load_llm_model_and_tokenizer(model_id: str):
    """
    Loads the LLM model and tokenizer from the given Hugging Face model ID.
    Returns (model, tokenizer) or (None, None) if loading fails.
    """
    logger.info(f"Attempting to load LLM model and tokenizer for: {model_id}")
    try:
        model, tokenizer = load(model_id)
        logger.info(f"Successfully loaded LLM model and tokenizer: {model_id}")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading LLM model {model_id}: {e}", exc_info=True)
        logger.error("Ensure the model ID is correct, the model is compatible with MLX, "
                     "and you have an internet connection if downloading for the first time.")
        logger.error("Common MLX models can be found under 'mlx-community' on Hugging Face.")
        return None, None

def format_gemma_chat_prompt(user_query: str, system_prompt: str = None) -> str:
    """
    Formats a prompt for Gemma instruction-tuned models using its chat template.
    Includes a basic system prompt if provided (though Gemma's primary training
    is more geared towards the user/model turn structure).
    """
    # Gemma's official chat template is typically:
    # <start_of_turn>user\n{user_query}<end_of_turn>\n<start_of_turn>model\n

    # System prompts are less formally defined for the base chat models but can be prepended.
    # For this implementation, we will prepend the system prompt if given, then follow
    # the user/model turn structure. Some variations might exist.
    # This template is a common interpretation.

    if system_prompt:
        # Note: Official Gemma guidance for system prompts in a multi-turn chat isn't as
        # explicit as for some other models. Often, the system context is part of the first user turn.
        # However, for single-turn API-like calls, prepending it before the user turn is a common strategy.
        # We will adopt a simple prepending here.
        # A more robust solution might involve inspecting tokenizer.chat_template if available and complex.
        # For now, we use this structure:
        # System Prompt (if any)
        # <start_of_turn>user
        # User Query
        # <end_of_turn>
        # <start_of_turn>model
        # (model response follows)

        # A simplified approach for single-turn tasks with a system message:
        # Prepend system message, then the standard user->model turn.
        # This is an area that might need adjustment based on observed Gemma behavior with mlx-lm.
        # For now, let's try:
        # system_prompt_formatted = f"<start_of_turn>system\n{system_prompt}<end_of_turn>\n" # Not standard Gemma
        # Let's stick to what's common: Prepend system prompt text before user turn.
        # It often gets implicitly included in the context by the model.

        # Gemma's preferred way for a system prompt is often to include it as part of the first user message.
        # Example from Google for single-turn system prompt usage:
        # prompt = f"Answer the following question: {FLAGS.question_with_system_prompt}" where question_with_system_prompt is "System prompt: What is the meaning of life?\n
        user_query_with_system = f"{system_prompt}\n\n{user_query}"
        formatted_prompt = f"<start_of_turn>user\n{user_query_with_system}<end_of_turn>\n<start_of_turn>model\n"
    else:
        formatted_prompt = f"<start_of_turn>user\n{user_query}<end_of_turn>\n<start_of_turn>model\n"
    
    return formatted_prompt

def invoke_llm_mlx(model, tokenizer, formatted_prompt: str, model_context_window: int,
                   max_tokens: int = 500, temperature: float = 0.7,
                   verbose_generation: bool = False) -> str | None:
    """
    Invokes the loaded MLX LLM. Checks if prompt fits context window.
    Returns the generated text, a specific error indicator string, or None for other errors.
    """
    logger.debug(f"invoke_llm_mlx called. Model context window: {model_context_window}, Max new tokens: {max_tokens}")

    if not tokenizer:
        logger.error("Tokenizer not provided to invoke_llm_mlx. Cannot proceed with token length check or generation.")
        return PROMPT_TOKENIZATION_FAILED_INDICATOR # Or a more specific "TOKENIZER_MISSING"

    try:
        prompt_token_ids = tokenizer.encode(formatted_prompt)
        prompt_token_count = len(prompt_token_ids)
        logger.info(f"Estimated prompt token count: {prompt_token_count}")
    except Exception as e:
        logger.error(f"Error tokenizing prompt for length check: {e}", exc_info=True)
        return PROMPT_TOKENIZATION_FAILED_INDICATOR

    # The prompt itself must fit within the model's context window.
    # `max_tokens` refers to the *newly generated* tokens, which also need space,
    # but the primary check here is for the input prompt. The model handles KV cache for generated tokens.
    if prompt_token_count >= model_context_window:
        logger.error(
            f"Prompt token count ({prompt_token_count}) exceeds or equals model context window ({model_context_window}). "
            f"Cannot process this request with the current model and prompt."
        )
        return INPUT_TOO_LONG_ERROR_INDICATOR

    # Safety check for Gemma prompt format (though format_gemma_chat_prompt should handle this)
    if not formatted_prompt.strip().endswith("<start_of_turn>model\n"):
        logger.warning("LLM prompt for Gemma did not end with '<start_of_turn>model\\n'. Appending it for safety.")
        formatted_prompt = formatted_prompt.rstrip() + "\n<start_of_turn>model\n"
        # Potentially re-tokenize and re-check length if this modification is significant,
        # but for a simple append, it's usually minor. For robustness, a re-check could be added.
        # For this plan, we'll assume this append doesn't critically alter length check outcome.

    logger.info(f"Invoking LLM. Effective prompt tokens: {prompt_token_count}. Max new tokens for generation: {max_tokens}, Temperature: {temperature}")
    logger.debug(f"Sending prompt to LLM (first 200 chars to avoid excessive logging): {formatted_prompt[:200]}...")
    
    try:
        # Import the proper sampler maker
        from mlx_lm.sample_utils import make_sampler
        
        logger.info(f"Creating sampler with temperature={temperature}")
        
        # Create a kwargs dict with the parameters we need
        gen_kwargs = {
            "max_tokens": max_tokens,
            "verbose": verbose_generation
        }
        
        # Use make_sampler to create a proper sampler function
        # For temperature > 0, we'll use a sampler with temperature and top_p
        if temperature > 0:
            sampler = make_sampler(temp=temperature, top_p=0.9)
            gen_kwargs["sampler"] = sampler
            logger.info(f"Using temperature sampling with temp={temperature}, top_p=0.9")
        else:
            # For temperature=0, make_sampler will return a greedy sampler (argmax)
            sampler = make_sampler(temp=0)
            gen_kwargs["sampler"] = sampler
            logger.info("Using greedy sampling (temperature=0)")
        
        logger.info(f"Generating with kwargs: {list(gen_kwargs.keys())}")
        
        # Generate the response
        response = generate(
            model,
            tokenizer,
            formatted_prompt,
            **gen_kwargs
        )
            
        logger.info("LLM generation complete.")
        logger.debug(f"LLM raw response (first 200 chars): {response[:200] if response else 'None'}")
        return response.strip() if response else None # Return None if LLM gives empty response
    except Exception as e:
        logger.error(f"Error during LLM generation: {e}", exc_info=True)
        # Crude check for common out-of-memory messages from MLX/Metal
        if "failed to allocate memory" in str(e).lower() or \
           "metal buffer could not be allocated" in str(e).lower() or \
           "out of memory" in str(e).lower():
            logger.error(
                "Potential out-of-memory error during LLM generation. "
                "This can happen if the prompt + KV cache for generated tokens exceeds available RAM/VRAM, "
                "even if the initial prompt token count was within the context window limit. "
                "Consider reducing max_tokens, using a smaller model, or ensuring more system memory is free."
            )
            return GENERATION_OOM_ERROR_INDICATOR
        return None # Indicate other general generation error