from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class GenerationArguments:
    """
    Arguments pertaining to specify the decoding parameters.
    """

    temperature: float = field(
        default=1.0,
        metadata={"help": "The value used to modulate the next token probabilities."},
    )
    top_p: float = field(
        default=1.0,
        metadata={
            "help": "The smallest set of most probable tokens with probabilities that add up to top_p or higher are kept."
        },
    )
    top_k: int = field(
        default=-1,
        metadata={
            "help": "The number of highest probability vocabulary tokens to keep for top-k filtering."
        },
    )
    num_beams: int = field(
        default=1,
        metadata={"help": "Number of beams for beam search. 1 means no beam search."},
    )
    max_length: int = field(
        default=1024,
        metadata={
            "help": "The maximum length the generated tokens can have. It can be overridden by max_new_tokens."
        },
    )
    max_new_tokens: int = field(
        default=16,
        metadata={
            "help": "The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt."
        },
    )
    repetition_penalty: float = field(
        default=1.0,
        metadata={
            "help": "The parameter for repetition penalty. 1.0 means no penalty."
        },
    )
    length_penalty: float = field(
        default=1.0,
        metadata={
            "help": "Exponential penalty to the length that is used with beam-based generation."
        },
    )
    default_system: Optional[str] = field(
        default=None,
        metadata={"help": "Default system message to use in chat completion."},
    )
    logprobs: int = field(
        default=0,
        metadata={
            "help": "The maximum length the generated tokens can have. It can be overridden by max_new_tokens."
        },
    )
    prompt_logprobs: int = field(
        default=0,
        metadata={
            "help": "The maximum length the generated tokens can have. It can be overridden by max_new_tokens."
        },
    )

    def to_dict(self) -> Dict[str, Any]:
        args = asdict(self)
        if args.get("max_new_tokens", -1) > 0:
            args.pop("max_length", None)
        else:
            args.pop("max_new_tokens", None)
        return args
