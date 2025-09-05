# just call analyze_trait_progression.py

import sys
import os

sys.path.append(os.path.dirname(__file__))

from example_finetuning_with_probes import analyze_trait_progression

analyze_trait_progression("trainer_output/trait_progression.json")