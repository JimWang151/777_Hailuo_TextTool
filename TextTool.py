import collections
import json
import os
import platform
import random
import time
import urllib.parse
import urllib.request
import uuid
import xml.etree.ElementTree as ET
from typing import List, Dict
import shutil

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
import subprocess
from torchvision.transforms import ToPILImage, ToTensor
from torchvision.transforms.functional import to_tensor, to_pil_image

class HL_TextToImage:

    def __init__(self):
        # 解析 XML 文件
        self.supported_font_extensions = [".ttf",".otf"]
        self.destination_font_dir = "/usr/share/fonts/"  # 硬编码字体安装目标目录

        # 获取当前目录下的 font 文件夹路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.fonts_dir = os.path.join(current_dir, "font")

        self.install_font_batch()  # 自动安装字体

    def validate_font_file(self, font_path):
        """
        验证字体文件路径和文件类型
        """
        if not os.path.exists(font_path):
            raise FileNotFoundError(f"字体文件 {font_path} 不存在！")

        file_extension = os.path.splitext(font_path)[1]
        if file_extension.lower() not in self.supported_font_extensions:
            raise ValueError(
                f"支持的字体类型为：{'、'.join(self.supported_font_extensions)}，当前文件为 '{file_extension}'！")

    def check_font_installed(self, font_path):
        """
        检查字体文件是否已安装
        """
        self.validate_font_file(font_path)  # 先验证字体文件

        font_file_name = os.path.basename(font_path)
        font_dirs = [self.destination_font_dir, os.path.expanduser("~/.fonts/")]  # 常见的字体目录
        for directory in font_dirs:
            if directory and os.path.exists(directory):
                if font_file_name in os.listdir(directory):
                    return True
        return False

    def install_font(self, font_path):
        """
        将字体文件安装到目标目录
        """
        self.validate_font_file(font_path)  # 验证字体文件

        destination_path = os.path.join(self.destination_font_dir, os.path.basename(font_path))

        # 检查目标目录是否存在，不存在则创建
        if not os.path.exists(self.destination_font_dir):
            os.makedirs(self.destination_font_dir)

        # 复制字体文件到目标目录
        shutil.copy2(font_path, destination_path)

        return destination_path

    @staticmethod
    def refresh_font_cache():
        """
        刷新字体缓存（跨平台）
        """
        system = platform.system()
        try:
            if system == "Linux":
                subprocess.run(["fc-cache", "-f"], capture_output=True, check=True)
                
            elif system == "Windows":
                subprocess.run(["powershell", "Start-Process", "C:\\Windows\\System32\\control.exe", "-ArgumentList",
                                "'C:\\Windows\\Fonts'"], check=True)
                
            else:
                print(f"不支持的操作系统：{system}，无法刷新字体缓存！")
        except subprocess.CalledProcessError as e:
            print(f"刷新字体缓存失败：{e}")

    def install_font_batch(self):
        """
        批量安装字体
        """
        if not os.path.exists(self.fonts_dir):
            raise FileNotFoundError(f"字体文件夹不存在：{self.fonts_dir}！")

        for font_file in os.listdir(self.fonts_dir):
            font_path = os.path.join(self.fonts_dir, font_file)
            if os.path.isfile(font_path) :
                try:
                    # 检查字体是否已安装
                    if not self.check_font_installed(font_path):
                        print(f"正在安装字体 {font_file} 到 {self.destination_font_dir} ...")
                        self.install_font(font_path)
                        
                except Exception as e:
                    print(f"字体 {font_file} 安装失败：{e}")

        self.refresh_font_cache()

    @classmethod
    def INPUT_TYPES(s):
        font_color_options = ["green", "red", "white", "black", "yellow", "orange", "cyan", "by color code"]
        return {
            "required": {
                "content": ("STRING", {"default": 'Trial Version'}),
                "font": (["Arial", "Times New Roman", "Courier New", "Verdana", "Tahoma", "SimSun", "SimHei", "Althelas", "AlthelasBold","Suite Home"],),
                "font_size": ("INT", {"default": 30, "min": 10, "max": 200, "step": 2}),
                "font_color": (font_color_options, {"default": 'black'}),  # 增加新选项
                "transparent": ("BOOLEAN", {"default": True}),
                "bg_color": (["green", "red", "white", "black", "yellow", "orange", "cyan"],),
                "max_chars_per_line": ("STRING", {"default": '100'}),
                "width": ("INT", {"default": 512, "min": 32, "max": 2048, "step": 1}),
                "height": ("INT", {"default": 512, "min": 32, "max": 2048, "step": 1}),
                "x_align": (["LEFT", "CENTER", "RIGHT"], {"default": 'CENTER'}),
                "y_align": (["TOP", "CENTER", "BOTTOM"], {"default": 'CENTER'}),
                "line_spacing": ("INT", {"default": 30, "min": 10, "max": 60, "step": 2}),
                "bold": ("BOOLEAN", {"default": False}),
                "rotation_angle": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.5}),
                "by_ratio": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "color_code": ("STRING", {"default": "#000000"}),  # 新增颜色代码参数
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("result_img", "content")
    FUNCTION = "txt_2_img"
    OUTPUT_NODE = True
    CATEGORY = "HL_Tools"
    DESCRIPTION = "Tools for generating text images"

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
		 "Suite Home": "Suite.otf",
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
                "Suite Home": "Suite.otf",
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

    def txt_2_img(self, content, font, font_size, font_color, transparent, bg_color,color_code, max_chars_per_line, width,
                  height, x_align, line_spacing, bold, y_align, by_ratio, rotation_angle):

        # 1. 字体处理
        font_path = self._find_font_file(font)
        if font_path is None:
            font = ImageFont.load_default()
        else:
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")

        # 2. 文本换行处理
        wrapped_lines = self._wrap_text(content, font, width - 40)  # 预留左右边距

        # 3. 计算文本尺寸
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

        # 4. 背景颜色处理
        if transparent:
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

        # 5. 创建图片
        img = Image.new("RGBA", (width, height), background_color)
        draw = ImageDraw.Draw(img)

        # 6. 文字颜色处理
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
            "gray": (196, 169, 169, 255),
            "cyan": (0, 255, 255, 255),
        }

        # 处理颜色选择逻辑
        if font_color == "by color code":
            # 使用用户输入的颜色代码
            try:
                text_fill = self.hex_to_rgba(color_code)
            except:
                # 如果颜色代码无效，使用黑色作为回退
                text_fill = (0, 0, 0, 255)
                print(f"警告：无效的颜色代码 '{color_code}'，已使用黑色替代")
        else:
            # 使用预定义的颜色映射
            text_fill = font_color_mapping.get(font_color, (0, 0, 0, 255))

        # 7. 计算边距（基于字体大小）
        margin = max(5, font_size // 6)  # 最小5像素，最大字体大小的1/6

        # 8. 计算垂直偏移量（y_offset）
        if by_ratio != 0.0:
            # 比例模式：按图像尺寸比例计算
            x_offset_base = int(width * by_ratio)
            y_offset = int(height * by_ratio)
        else:
            # 传统对齐模式
            # 处理文本高度大于画布的情况
            if total_text_height > height:
                y_offset = -margin
                print(f"警告：文本高度({total_text_height}px)超过画布高度({height}px)，部分文本可能不可见")
            else:
                # 根据垂直对齐方式计算y_offset
                if y_align == "CENTER":
                    y_offset = (height - total_text_height) // 2
                elif y_align == "TOP":
                    y_offset = margin
                elif y_align == "BOTTOM":
                    y_offset = height - total_text_height - margin
                else:  # 默认为居中
                    y_offset = (height - total_text_height) // 2

        # 9. 绘制每行文本
        for line in wrapped_lines:
            # 计算当前行尺寸
            text_bbox = font.getbbox(line)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            # 计算当前行的水平偏移量（x_offset_line）
            if by_ratio != 0.0:
                # 比例模式：使用统一的基础偏移量
                x_offset_line = x_offset_base
            else:
                # 处理当前行宽度大于画布的情况
                if text_width > width:
                    x_offset_line = -margin
                    print(f"警告：行文本宽度({text_width}px)超过画布宽度({width}px)，部分文本可能被裁剪")
                else:
                    # 根据水平对齐方式计算x_offset_line
                    if x_align == "LEFT":
                        x_offset_line = margin
                    elif x_align == "CENTER":
                        x_offset_line = (width - text_width) // 2
                    elif x_align == "RIGHT":
                        x_offset_line = width - text_width - margin
                    else:  # 默认为左对齐
                        x_offset_line = margin

            # 绘制文本（支持加粗）
            self._draw_bold_text(draw, x_offset_line, y_offset, line, font, text_fill, bold)

            # 更新垂直位置（下一行）
            y_offset += text_height + line_spacing

        # 10. 应用旋转（如果有）
        if rotation_angle != 0:
            img = self._apply_rotation(img, rotation_angle, transparent, background_color)

        # 11. 转换为PyTorch Tensor
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).unsqueeze(0)

        return (img_tensor, content)

    def _apply_rotation(self, image, angle, is_transparent, background_color):
        """
        将图像旋转指定角度

        参数:
            image: PIL.Image 对象
            angle: 旋转角度（度）
            is_transparent: 背景是否透明
            background_color: 原始背景颜色

        返回:
            旋转后的 PIL.Image 对象
        """
        # 确定旋转后的填充色
        if is_transparent:
            fill_color = (0, 0, 0, 0)  # 透明填充
        else:
            # 处理背景色格式
            if len(background_color) == 4:  # 已经是RGBA
                fill_color = background_color
            else:  # RGB格式
                fill_color = background_color + (255,)  # 添加不透明通道

        # 执行旋转
        return image.rotate(
            angle,
            resample=Image.BICUBIC,  # 使用高质量插值
            expand=False,  # 保持原始尺寸
            fillcolor=fill_color  # 旋转后空白区域的填充色
        )


    def _draw_bold_text(self, draw, x, y, text, font, fill, bold):
        """绘制加粗文本"""
        if bold:
            offsets = [(0, 0), (1, 0), (0, 1), (1, 1)]
            for dx, dy in offsets:
                draw.text((x + dx, y + dy), text, font=font, fill=fill)
        else:
            draw.text((x, y), text, font=font, fill=fill)

    def _apply_rotation(self, image, angle, is_transparent, background_color):
        """
        将图像旋转指定角度

        参数:
            image: PIL.Image 对象
            angle: 旋转角度（度）
            is_transparent: 背景是否透明
            background_color: 原始背景颜色

        返回:
            旋转后的 PIL.Image 对象
        """
        # 确定旋转后的填充色
        if is_transparent:
            fill_color = (0, 0, 0, 0)  # 透明填充
        else:
            # 处理背景色格式
            if len(background_color) == 4:  # 已经是RGBA
                fill_color = background_color
            else:  # RGB格式
                fill_color = background_color + (255,)  # 添加不透明通道

        # 执行旋转
        return image.rotate(
            angle,
            resample=Image.BICUBIC,  # 使用高质量插值
            expand=False,  # 保持原始尺寸
            fillcolor=fill_color  # 旋转后空白区域的填充色
        )

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


import random
from typing import List, Dict, Any

class ZodiacPromptGenerator:
    """生肖提示词生成器 - ComfyUI 插件"""

    @classmethod
    def INPUT_TYPES(cls):
        """定义输入参数"""
        return {
            "required": {
                "human_gender": ("STRING", {
                    "multiline": False,
                    "default": "man",
                    "display": "人物性别"
                }),
                "base_seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "display": "基础种子"
                }),
            },
            "optional": {
                "human_desc": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "display": "人物描述"
                }),
            }
        }

    RETURN_TYPES = ("JOB", "INT", "INT[]", "STRING[]", "STRING[]")
    RETURN_NAMES = ("prompt_collections", "seed_no", "seed_list", "prompt_list", "zodiac_list")
    FUNCTION = "generate_zodiac_prompts"
    CATEGORY = "Prompts/Zodiac"

    def generate_zodiac_prompts(self, human_gender: str, base_seed: int, human_desc: str = "") -> tuple[List[Dict[str, Any]], int, List[int], List[str], List[str]]:
        """
        生成12个包含生肖、动作和种子的提示词集合，以及种子、提示词和生肖列表

        参数:
            human_gender: 提示词中的人物性别（例如 'man' 或 'woman'）
            base_seed: 种子生成的基础值
            human_desc: 人物描述（可选），若不为空则替换默认描述

        返回:
            prompt_collections: 包含提示词和种子的集合
            seed_no: 最后一个使用的种子值
            seed_list: 所有种子值的列表
            prompt_list: 所有提示词的列表
            zodiac_list: 生肖名称列表
        """
        # 生肖列表，按指定顺序
        zodiac_map = [
            {"en": "zodiac_mouse", "name": "Mouse"},
            {"en": "zodiac_cow", "name": "Cow"},
            {"en": "zodiac_tiger", "name": "Tiger"},
            {"en": "zodiac_rabbit", "name": "Rabbit"},
            {"en": "zodiac_dragon", "name": "Dragon"},
            {"en": "zodiac_snake", "name": "Snake"},
            {"en": "zodiac_horse", "name": "Horse"},
            {"en": "zodiac_sheep", "name": "Sheep"},
            {"en": "zodiac_monkey", "name": "Monkey"},
            {"en": "zodiac_chicken", "name": "Chicken"},
            {"en": "zodiac_dog", "name": "Dog"},
            {"en": "zodiac_pig", "name": "Pig"}
        ]

        # 动作列表
        action_list = [
            'is kissing by',
            'is holding by',
            'is standing on the left shoulder of',
            'is standing on the right shoulder of',
            'is playing with'
        ]

        # 固定的提示词后缀
        prompt_suffix = ", clear and detailed hands."

        # 默认人物描述
        default_desc = f"the {human_gender} with short black hair and a cheerful expression"

        # 使用提供的描述或默认描述
        human_description = human_desc if human_desc else default_desc

        # 初始化结果集合
        prompt_collections = []
        seed_list = []
        prompt_list = []
        zodiac_list = [zodiac["name"] for zodiac in zodiac_map]

        # 设置随机种子以确保可重复性
        random.seed(base_seed)

        try:
            # 生成12个提示词
            for zodiac in zodiac_map:
                # 随机选择一个动作
                action = random.choice(action_list)
                # 构建完整的提示词
                whole_prompt = (f"zodiac_cn, {zodiac['en']}, "
                               f"the animal named {zodiac['en']} {action} a {human_gender}, "
                               f"{human_description}{prompt_suffix}")
                # 生成种子
                current_seed = random.randint(0, 0xffffffffffffffff)
                # 添加到集合
                prompt_collections.append({
                    "prompt": whole_prompt,
                    "seed_no": current_seed
                })
                seed_list.append(current_seed)
                prompt_list.append(whole_prompt)

            print(f"✅ 成功生成 {len(prompt_collections)} 个提示词")
            return (prompt_collections, current_seed, seed_list, prompt_list, zodiac_list)

        except Exception as e:
            print(f"❌ 生成提示词失败: {str(e)}")
            return (prompt_collections, base_seed, seed_list, prompt_list, zodiac_list)


from typing import List, Union

class SelFromList:
    """从列表中选择单个元素 - ComfyUI 插件"""

    @classmethod
    def INPUT_TYPES(cls):
        """定义输入参数"""
        return {
            "required": {
                "input_list": ("ANY",),
                "select": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 99999,
                    "step": 1,
                    "display": "选择索引"
                }),
            },
        }

    RETURN_TYPES = ("ANY",)
    RETURN_NAMES = ("selected_item",)
    INPUT_IS_LIST = (True,)
    FUNCTION = "select"
    CATEGORY = "Prompts/Zodiac"

    def select(self, input_list: List[Union[int, str]], select: List[int]) -> tuple[Union[int, str]]:
        """
        根据索引从输入列表中选择单个元素

        参数:
            input_list: 输入的列表（可以是整数或字符串）
            select: 要选择的索引

        返回:
            selected_item: 选中的单个元素
        """
        select_idx = select[0]  # 获取索引值
        n = len(input_list)

        # 如果索引超出范围，选择最后一个元素
        if select_idx >= n:
            select_idx = n - 1

        try:
            return (input_list[select_idx],)
        except Exception as e:
            print(f"❌ 选择元素失败: {str(e)}")
            return (None,)




