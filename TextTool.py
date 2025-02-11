# Made by JimWang for ComfyUI
# 02/04/2023
import collections
import json
import os
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

    def _find_font_file(self, font_name):
        """查找字体文件"""
        windows_font_path = "C:/Windows/Fonts"
        font_mapping = {
            "Arial": "arial.ttf",
            "Times New Roman": "times.ttf",
            "Courier New": "cour.ttf",
            "Verdana": "verdana.ttf",
            "Tahoma": "tahoma.ttf",
            "SimSun": "simsun.ttc",
            "SimHei": "simhei.ttf",
        }
        if font_name in font_mapping:
            font_file = font_mapping[font_name]
            font_path = os.path.join(windows_font_path, font_file)
            if os.path.exists(font_path):
                return font_path
        return None

    def _wrap_text(self, text, max_chars_per_line):
        """按最大字符数换行"""
        lines = []
        current_line = ""
        for char in text:
            if len(current_line) + len(char) <= max_chars_per_line:
                current_line += char
            else:
                lines.append(current_line)
                current_line = char
        if current_line:
            lines.append(current_line)
        return lines

    def txt_2_img(self, seed, content, font, font_size, font_color, transparent, bg_color, max_chars_per_line, width, height, align, line_spacing, bold):
        """生成文字图片"""

        try:
            font_size = int(font_size)
            max_chars_per_line = int(max_chars_per_line)
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
        wrapped_lines = self._wrap_text(content, max_chars_per_line)

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
            }
            background_color = color_mapping.get(bg_color, (255, 255, 255))

        # 创建图片
        img = Image.new("RGBA", (width, height), background_color)
        draw = ImageDraw.Draw(img)

        # 文字颜色
        font_color_mapping = {
            "black": (0, 0, 0, 255),
            "red": (255, 0, 0, 255),
            "green": (0, 255, 0, 255),
            "blue": (0, 0, 255, 255),
            "white": (255, 255, 255, 255),
        }
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
            elif align == 3:  # 两端对齐
                words = line.split()
                if len(words) > 1:
                    total_word_width = sum(font.getbbox(word)[2] - font.getbbox(word)[0] for word in words)
                    extra_space = (width - total_word_width - 40) // (len(words) - 1)
                    x_offset = 20
                    for word in words:
                        self._draw_bold_text(draw, x_offset, y_offset, word, font, text_fill, bold)
                        x_offset += font.getbbox(word)[2] - font.getbbox(word)[0] + extra_space
                    y_offset += text_height + line_spacing
                    continue
                else:
                    x_offset = 20

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






