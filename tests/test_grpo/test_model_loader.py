import pytest
from unittest.mock import patch, MagicMock
import sys

from models.model_loader import load_model_for_training

def test_load_model_for_training_with_adapter_path():
    # Setup sys.modules mocks to bypass local imports
    mock_unsloth = MagicMock()
    mock_peft = MagicMock()
    
    mock_base_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_unsloth.FastVisionModel.from_pretrained.return_value = (mock_base_model, mock_tokenizer)
    
    mock_peft_model = MagicMock()
    mock_peft.PeftModel.from_pretrained.return_value = mock_peft_model
    
    with patch.dict(sys.modules, {"unsloth": mock_unsloth, "peft": mock_peft}):
    
        # Run the function
        adapter_path = "/path/to/my/adapter"
        model, tokenizer, info = load_model_for_training(
            model_name="Qwen/Qwen2-VL-2B",
            tier="2b",
            sft_cfg={},
            adapter_path=adapter_path
        )
        
        # Verify FastVisionModel was called with the base model name
        mock_unsloth.FastVisionModel.from_pretrained.assert_called_once()
        assert mock_unsloth.FastVisionModel.from_pretrained.call_args[0][0] == "Qwen/Qwen2-VL-2B"
        
        # Verify PeftModel was called with the base model and the adapter path
        mock_peft.PeftModel.from_pretrained.assert_called_once_with(mock_base_model, adapter_path, is_trainable=True)
        
        # Verify the returned model is the PeftModel
        assert model == mock_peft_model
