# Made by Jim.Wang V1 for ComfyUI
import os
import subprocess
import importlib.util
import sys
import filecmp
import shutil

import __main__

python = sys.executable




from .TextTool import HL_TextToImage,HL_FilterImage,ChiikawaPromptGenerator,SelFromList

NODE_CLASS_MAPPINGS = {
    "HL_TextToImage":HL_TextToImage,
    "HL_FilterImage":HL_FilterImage,
"ChiikawaPromptGenerator":ChiikawaPromptGenerator,
    "SelFromList":SelFromList
}


print('\033[34mHailuo TextTool Nodes: \033[92mLoaded\033[0m')