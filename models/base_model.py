"""Base model for CL-CoT"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from typing import Optional, Dict, Any


class CLCoTModel(nn.Module):
    """CL-CoT base model with LoRA"""

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-13b-hf",
        use_lora: bool = True,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: list = None,
        device: str = "cuda"
    ):
        """
        Initialize CL-CoT model

        Args:
            model_name: Pretrained model name
            use_lora: Whether to use LoRA
            lora_rank: LoRA rank
            lora_alpha: LoRA alpha
            lora_dropout: LoRA dropout
            lora_target_modules: Target modules for LoRA
            device: Device to use
        """
        super().__init__()

        self.model_name = model_name
        self.use_lora = use_lora
        self.device = device

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )

        # Apply LoRA if requested
        if use_lora:
            if lora_target_modules is None:
                lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
                bias="none"
            )

            self.model = get_peft_model(self.base_model, lora_config)
            print(f"LoRA applied with rank={lora_rank}, alpha={lora_alpha}")
            print(f"Trainable parameters: {self.model.print_trainable_parameters()}")
        else:
            self.model = self.base_model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Labels for training

        Returns:
            Dictionary with loss and logits
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        return {
            'loss': outputs.loss if labels is not None else None,
            'logits': outputs.logits
        }

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        num_beams: int = 1,
        do_sample: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate text

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            num_beams: Number of beams for beam search
            do_sample: Whether to use sampling
            **kwargs: Additional generation arguments

        Returns:
            Generated token IDs
        """
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_beams=num_beams,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )

        return outputs

    def save_pretrained(self, save_directory: str):
        """Save model"""
        if self.use_lora:
            self.model.save_pretrained(save_directory)
        else:
            self.base_model.save_pretrained(save_directory)

        self.tokenizer.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: str = "cuda"
    ) -> 'CLCoTModel':
        """Load pretrained model"""
        # This is simplified - in practice would need to load config
        model = cls(model_name=model_path, device=device)
        return model

    def to(self, device: str):
        """Move model to device"""
        self.model = self.model.to(device)
        self.device = device
        return self

    def train(self):
        """Set to training mode"""
        self.model.train()

    def eval(self):
        """Set to evaluation mode"""
        self.model.eval()

    def get_trainable_parameters(self) -> int:
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
