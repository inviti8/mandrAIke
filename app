import os
from pathlib import Path
import click
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import random

from MandrAIk import *


BRAND = "HEAVYMETA®"
VERSION = "0.01"
ABOUT = f"""
Command Line Interface for {BRAND} mandrAIk 
Version: {VERSION}
ALL RIGHTS RESERVED 2025
"""
VERSION = "0.01"

FILE_PATH = Path(__file__).parent
HOME = os.path.expanduser('~')
CLI_PATH = os.path.join(HOME, '.local', 'share', 'mandrAIk')




class MandrAIkGroup(click.Group):
    def parse_args(self, ctx, args):
        if args[0] in self.commands:
            if len(args) == 1 or args[1] not in self.commands:
                args.insert(0, '')
        super(MandrAIkGroup, self).parse_args(ctx, args)


@click.command('poison')
@click.argument('in_img_path', type=str)
@click.argument('out_img_path', type=str)
def poison(in_img_path, out_img_path):
      """Add Mandrake Poison to Image"""
      if os.path.isfile(in_img_path):
            adversarial = MandrAIk()
