# just call analyze_trait_progression.py

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from probes.examples.example_finetuning_with_probes import analyze_trait_progression

analyze_trait_progression("trainer_output/snake_trait_progression.json")