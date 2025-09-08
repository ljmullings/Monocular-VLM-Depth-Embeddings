"""PS3 (Progressive Sampling Strategy) controls for VILA."""

import os
from typing import Tuple


class PS3Controller:
    """Controller for VILA PS3 parameters."""
    
    def __init__(
        self,
        num_look_close: int = 2,
        num_token_look_close: int = 2048,
        select_num_each_scale: str = "256+512",
    ):
        """
        Initialize PS3 controller.
        
        Args:
            num_look_close: Number of high-res patches
            num_token_look_close: Tokens per high-res patch
            select_num_each_scale: Token allocation per scale
        """
        self.num_look_close = num_look_close
        self.num_token_look_close = num_token_look_close
        self.select_num_each_scale = select_num_each_scale
        
        # Set environment variables
        self.set_environment()
    
    def set_environment(self) -> None:
        """Set PS3 environment variables."""
        os.environ["NUM_LOOK_CLOSE"] = str(self.num_look_close)
        os.environ["NUM_TOKEN_LOOK_CLOSE"] = str(self.num_token_look_close)
        os.environ["SELECT_NUM_EACH_SCALE"] = self.select_num_each_scale
    
    @classmethod
    def from_environment(cls) -> "PS3Controller":
        """Create controller from environment variables."""
        return cls(
            num_look_close=int(os.getenv("NUM_LOOK_CLOSE", 2)),
            num_token_look_close=int(os.getenv("NUM_TOKEN_LOOK_CLOSE", 2048)),
            select_num_each_scale=os.getenv("SELECT_NUM_EACH_SCALE", "256+512"),
        )
    
    def get_token_budget(self) -> Tuple[int, int]:
        """Calculate total token budget."""
        # Parse select_num_each_scale
        scales = self.select_num_each_scale.split("+")
        total_base_tokens = sum(int(s) for s in scales)
        total_high_res_tokens = self.num_look_close * self.num_token_look_close
        
        return total_base_tokens, total_high_res_tokens
