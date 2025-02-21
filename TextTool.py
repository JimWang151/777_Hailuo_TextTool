# Made by JimWang for ComfyUI
# 02/04/2023
import collections
import json
import os
import os
import platform
import random
import time
import urllib.parse
import urllib.request
import uuid
import xml.etree.ElementTree as ET
from typing import List, Dict

import comfy.utils
import folder_paths
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
from PIL import Image, ImageOps, ImageSequence, ImageFile
from PIL.PngImagePlugin import PngInfo
from comfy.cli_args import args
from lxml import etree
from torchvision.transforms import ToPILImage, ToTensor
from torchvision.transforms.functional import to_tensor, to_pil_image


import platform
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch

class HL_TextToImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                 "tooltip": "The random seed used for creating the noise."}),
                "content": ("STRING", {"default": 'Trial Version'}),
                "font": ("STRING", {"default": 'Courier New'}),
                "font_size": ("STRING", {"default": '40'}),
                "font_color": ("STRING", {"default": 'black'}),
                "transparent": ("STRING", {"default": '1'}),
                "bg_color": ("STRING", {"default": ''}),
                "max_chars_per_line": ("STRING", {"default": '100'}),
                "width": ("STRING", {"default": '400'}),
                "height": ("STRING", {"default": '200'}),
                "align": ("STRING", {"default": '1'}),  # 1 左对齐，2 居中对齐，3 两端对齐
                "line_spacing": ("STRING", {"default": '10'}),  # 行间距
                "bold": ("STRING", {"default": '0'})  # 是否加粗 (0: 否, 1: 是)
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("result_img", "content")
    FUNCTION = "txt_2_img"
    OUTPUT_NODE = True
    CATEGORY = "HL_Tools"
    DESCRIPTION = "Tools for generating text images"

    # 新增颜色转换方法
    def hex_to_rgba(self, hex_color):
        """将十六进制颜色代码转换为 RGBA 元组，支持 #ABC 和 #AABBCC 格式"""
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 3:  # 如果是 #RGB 格式，扩展为 #RRGGBB
            hex_color = ''.join([c * 2 for c in hex_color])

        if len(hex_color) != 6:
            raise ValueError("Invalid hex color format")

        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return (r, g, b, 255)  # 默认透明度为 255（不透明）

    def _find_font_file(self, font_name):
        """根据操作系统查找字体文件"""

        system_name = platform.system()

        if system_name == "Windows":
            font_paths = ["C:/Windows/Fonts"]
            font_mapping = {
                "Arial": "arial.ttf",
                "Times New Roman": "times.ttf",
                "Courier New": "cour.ttf",
                "Verdana": "verdana.ttf",
                "Tahoma": "tahoma.ttf",
                "SimSun": "simsun.ttc",
                "SimHei": "simhei.ttf",
                "Althelas": "Athelas-Regular.ttf",
                "AlthelasBold": "Athelas-Bold.ttf",
            }

        elif system_name == "Linux":
            font_paths = [
                "/usr/share/fonts/truetype",
                "/usr/share/fonts",
                "/usr/share/fonts/truetype/dejavu",
                "~/.fonts",
            ]
            font_mapping = {
                "Arial": "arial.ttf",
                "Times New Roman": "times.ttf",
                "Courier New": "cour.ttf",
                "Verdana": "verdana.ttf",
                "Tahoma": "tahoma.ttf",
                "Noto Sans": "NotoSans-Regular.ttf",
                "DejaVu Sans": "DejaVuSans.ttf",
                "Althelas": "Athelas-Regular.ttf",
                "AlthelasBold": "Athelas-Bold.ttf",
            }

        else:
            return None  # 其他系统暂不支持

        # 搜索字体文件
        if font_name in font_mapping:
            font_file = font_mapping[font_name]
            for font_path in font_paths:
                full_path = os.path.join(font_path, font_file)
                if os.path.exists(full_path):
                    return full_path

        return None  # 找不到字体

    def _wrap_text(self, text, font, max_width):
        """按单词换行，确保单词不会被截断"""
        words = text.split()
        lines = []
        current_line = ""

        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            text_width = font.getbbox(test_line)[2] - font.getbbox(test_line)[0]

            if text_width <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word  # 新行从当前单词开始

        if current_line:
            lines.append(current_line)

        return lines

    def txt_2_img(self, seed, content, font, font_size, font_color, transparent, bg_color, max_chars_per_line, width, height, align, line_spacing, bold):
        """生成文字图片"""

        try:
            font_size = int(font_size)
            width = int(width)
            height = int(height)
            align = int(align)
            line_spacing = int(line_spacing)
            bold = bool(int(bold))  # 转换为布尔值
        except ValueError:
            return None, "Invalid input parameters"

        # 查找字体文件
        font_path = self._find_font_file(font)
        if font_path is None:
            font = ImageFont.load_default()
        else:
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")

        # 换行文本
        wrapped_lines = self._wrap_text(content, font, width - 40)  # 预留左右边距

        # 计算文本尺寸
        temp_image = Image.new("RGBA", (1, 1))
        temp_draw = ImageDraw.Draw(temp_image)
        line_heights = []
        max_line_width = 0

        for line in wrapped_lines:
            text_bbox = font.getbbox(line)
            line_width = text_bbox[2] - text_bbox[0]
            line_height = text_bbox[3] - text_bbox[1]
            line_heights.append(line_height)
            if line_width > max_line_width:
                max_line_width = line_width

        total_text_height = sum(line_heights) + (len(wrapped_lines) - 1) * line_spacing

        # 背景颜色
        if transparent == "1":
            background_color = (0, 0, 0, 0)  # 透明背景
        else:
            color_mapping = {
                "green": (0, 255, 0),
                "red": (255, 0, 0),
                "white": (255, 255, 255),
                "black": (0, 0, 0),
                "yellow": (255, 255, 0),
                "orange": (255, 165, 0),
                "cyan": (0, 255, 255),
            }
            background_color = color_mapping.get(bg_color, (255, 255, 255))

        # 创建图片
        img = Image.new("RGBA", (width, height), background_color)
        draw = ImageDraw.Draw(img)

        # 文字颜色
        #font_color = "black"
        font_color_mapping = {
            "black": (0, 0, 0, 255),
            "red": (255, 0, 0, 255),
            "green": (0, 255, 0, 255),
            "blue": (0, 0, 255, 255),
            "white": (255, 255, 255, 255),
            "yellow": (255, 255, 0, 255),
            "purple": (128, 0, 128, 255),
            "orange": (255, 165, 0, 255),
            "pink": (255, 192, 203, 255),
            "brown": (139, 69, 19, 255),
            "gray": (169, 169, 169, 255),
            "cyan": (0, 255, 255, 255),
        }

        if font_color.startswith('#'):
            # 如果是以 # 号开头的颜色代码，则调用 color_mapping 方法
            try:
                text_fill = self.hex_to_rgba(font_color)
            except:
                # 如果颜色代码无效，使用默认颜色
                text_fill = (0, 0, 0, 255)
        else:
            # 使用预定义的颜色名称
            text_fill = font_color_mapping.get(font_color, (0, 0, 0, 255))

        # 文字绘制起始位置
        y_offset = (height - total_text_height) // 2

        for line in wrapped_lines:
            text_bbox = font.getbbox(line)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            if align == 1:  # 左对齐
                x_offset = 20
            elif align == 2:  # 居中对齐
                x_offset = (width - text_width) // 2

            self._draw_bold_text(draw, x_offset, y_offset, line, font, text_fill, bold)
            y_offset += text_height + line_spacing

        # 转换为PyTorch Tensor
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).unsqueeze(0)

        return (img_tensor, content)

    def _draw_bold_text(self, draw, x, y, text, font, fill, bold):
        """绘制加粗文本"""
        if bold:
            offsets = [(0, 0), (1, 0), (0, 1), (1, 1)]
            for dx, dy in offsets:
                draw.text((x + dx, y + dy), text, font=font, fill=fill)
        else:
            draw.text((x, y), text, font=font, fill=fill)


class HL_FilterImage:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "img_url": ("STRING", {"multiline": True, "tooltip": "Key car."}),
            },
        }

    RETURN_TYPES = ("IMAGE","MASK")
    RETURN_NAMES = ("ORG_IMAGE","mask")
    FUNCTION = "Filter_Image"
    OUTPUT_NODE = True
    CATEGORY = "HL_Tools"
    DESCRIPTION = "Filter and mask"

    def Filter_Image(self, img_url):
        # 确保 img 是一个 4 通道张量（RGBA）
        img = Image.open(img_url)  # 直接打开图像

        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']

        for i in ImageSequence.Iterator(img):
            i = ImageOps.exif_transpose(i)  # 处理EXIF旋转信息

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                continue

            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask)






