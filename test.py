# -*- coding: utf-8 -*- # Add utf-8 encoding hint
import sys
import os
import torch
import torchaudio
import glob # 用于查找文件
import re   # 用于文本处理
import types # 用于检查 generator 类型
import traceback # 用于打印详细错误信息
import gc # 用于垃圾回收

# --- 配置区 ---
# 如果你的目录结构不同，请修改这些路径
MATCHA_TTS_PATH = 'third_party/Matcha-TTS' # Matcha-TTS 代码库路径
MODEL_PATH = 'pretrained_models/CosyVoice2-0.5B' # 预训练模型路径
ASSET_DIR = './asset' # 存放音色提示音频 (.wav) 的目录
OUTPUT_DIR = './output' # 输出音频保存目录
# --- 配置区结束 ---

# 将 Matcha-TTS 添加到 Python 路径
if MATCHA_TTS_PATH not in sys.path:
    sys.path.append(MATCHA_TTS_PATH)

# 尝试导入 CosyVoice 模块
try:
    # 假设这些导入是正确的
    from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
    from cosyvoice.utils.file_utils import load_wav
    # 尝试导入日志记录器，以便我们可以控制其级别（可选）
    import logging
    # 设置 CosyVoice 及相关库的日志级别，减少冗余输出
    # logging.getLogger('cosyvoice').setLevel(logging.WARNING)
    # logging.getLogger('funasr').setLevel(logging.ERROR)
    # logging.getLogger('modelscope').setLevel(logging.ERROR)
    # 在调试时可以注释掉以上行，查看所有日志
    pass # 保持默认日志级别，以便查看详细信息
except ImportError as e:
    print(f"错误：无法导入 CosyVoice 模块。")
    print(f"请确认 '{MATCHA_TTS_PATH}' 路径正确，并且已按照说明安装所有依赖。")
    print(f"详细错误: {e}")
    sys.exit(1)
except Exception as e:
     print(f"导入 CosyVoice 相关模块时发生其他错误: {e}")
     pass # 可能只是日志记录器设置失败，不一定退出

def load_model():
    """加载 CosyVoice 模型"""
    if not os.path.exists(MODEL_PATH):
        print(f"错误：找不到模型目录 '{MODEL_PATH}'")
        print("请确保预训练模型已下载并放置在正确的位置。")
        sys.exit(1)

    # 在加载模型前先清理内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    print(f"正在从 '{MODEL_PATH}' 加载 CosyVoice 模型...")
    print("这可能需要一些时间，特别是首次加载时。")
    try:
        # 使用内存优化选项加载模型
        # 设置 fp16=True 可以减少内存占用（如果您的设备支持）
        # 如果内存仍然不足，可以考虑设置 load_jit=True
        cosyvoice = CosyVoice2(
            MODEL_PATH, 
            load_jit=False, 
            load_trt=False, 
            fp16=False  # 如果您的设备支持，可以设为 True 以减少内存占用
            # 注意：移除了 text_frontend 参数，因为当前版本不支持
        )
        print("模型加载成功！")
        # 尝试获取并打印采样率
        sample_rate = getattr(cosyvoice, 'sample_rate', '未知')
        print(f"模型预期采样率: {sample_rate}")
        if sample_rate == '未知':
             print("警告：无法从模型对象获取采样率，将使用默认值保存音频。")
        return cosyvoice
    except NameError:
        print("错误：CosyVoice2 类未找到。请检查 cosyvoice.cli.cosyvoice 导入。")
        sys.exit(1)
    except Exception as e:
        print(f"错误：加载 CosyVoice 模型失败: {e}")
        print("请检查您的安装、模型文件、硬件兼容性（例如 GPU 驱动程序）。")
        traceback.print_exc() # 打印详细错误信息
        sys.exit(1)

def list_and_select_prompt_audio(asset_dir):
    """列出 asset 目录中的 .wav 文件并让用户选择"""
    print("\n--- 选择音色/风格提示音频 ---")
    if not os.path.isdir(asset_dir):
        print(f"错误：找不到音色资源目录 '{asset_dir}'")
        return None, None

    wav_files = sorted(glob.glob(os.path.join(asset_dir, '*.wav')))

    if not wav_files:
        print(f"错误：在 '{asset_dir}' 目录中没有找到任何 .wav 文件作为音色提示。")
        return None, None

    # 检查每个文件的长度，标记超过30秒的文件
    wav_files_info = []
    for f_path in wav_files:
        filename = os.path.basename(f_path)
        try:
            audio_info = torchaudio.info(f_path)
            duration_seconds = audio_info.num_frames / audio_info.sample_rate
            is_too_long = duration_seconds > 30
            wav_files_info.append({
                'path': f_path,
                'filename': filename,
                'duration': duration_seconds,
                'is_too_long': is_too_long
            })
        except Exception as e:
            print(f"警告: 无法获取 {filename} 的信息: {e}")
            wav_files_info.append({
                'path': f_path,
                'filename': filename,
                'duration': 0,
                'is_too_long': False  # 默认不标记为过长
            })

    print("可用的音色提示音频:")
    for i, info in enumerate(wav_files_info):
        duration_str = f"{info['duration']:.1f}秒" if info['duration'] > 0 else "未知时长"
        warning = " [警告: 超过30秒]" if info['is_too_long'] else ""
        print(f"{i + 1}. {info['filename']} ({duration_str}){warning}")

    while True:
        try:
            # 选择一个不超过30秒的默认文件
            default_index = 0
            for i, info in enumerate(wav_files_info):
                if not info['is_too_long']:
                    default_index = i
                    break
            
            default_filename = wav_files_info[default_index]['filename']
            choice_prompt = f"请输入选项编号 (1-{len(wav_files_info)}) 或直接按 Enter 使用默认 ({default_filename}): "
            choice = input(choice_prompt)
            if not choice:
                selected_index = default_index
                break
            selected_index = int(choice) - 1
            if 0 <= selected_index < len(wav_files_info):
                break
            else:
                print(f"无效的选择，请输入 1 到 {len(wav_files_info)} 之间的编号。")
        except ValueError:
            print("无效的输入，请输入数字。")

    selected_info = wav_files_info[selected_index]
    selected_path = selected_info['path']
    selected_filename = selected_info['filename']
    selected_filename_no_ext = os.path.splitext(selected_filename)[0]
    
    # 检查所选文件是否超过30秒
    if selected_info['is_too_long']:
        duration_str = f"{selected_info['duration']:.1f}秒" if selected_info['duration'] > 0 else "未知时长"
        print(f"警告: 您选择的音频 '{selected_filename}' 超过30秒 ({duration_str})。")
        print("这可能导致合成失败，因为模型不支持处理超过30秒的音频。")
        confirm = input("是否仍然继续? (y/n, 默认n): ").lower().strip()
        if confirm != 'y':
            print("请重新运行程序并选择其他音频文件。")
            sys.exit(0)
        print("尝试使用长音频文件，可能会出错...")
    else:
        duration_str = f"{selected_info['duration']:.1f}秒" if selected_info['duration'] > 0 else "未知时长"
        print(f"已选择: {selected_filename} ({duration_str})")

    try:
        print(f"正在加载提示音频: {selected_path}...")
        # 强制指定目标采样率为16000Hz，这是很多模型需要的
        prompt_speech_16k = load_wav(selected_path, 16000)
        if prompt_speech_16k is None or prompt_speech_16k.numel() == 0:
             raise ValueError("加载的音频为空或无效。")
        print(f"提示音频加载成功 (Shape: {prompt_speech_16k.shape}, Sample Rate: 16000 Hz)。")
        return prompt_speech_16k, selected_filename_no_ext
    except Exception as e:
        print(f"错误：加载提示音频 '{selected_path}' 失败: {e}")
        traceback.print_exc()
        return None, None

def get_user_input(prompt_message):
    """获取用户输入，支持多行"""
    print(f"{prompt_message} (输入完成后，空行处按 Enter 结束):")
    lines = []
    try:
        while True:
            # 尝试使用更健壮的方式读取，处理潜在的编码问题
            line = sys.stdin.readline().rstrip('\r\n')
            if line:
                lines.append(line)
            else:
                break
    except EOFError:
        pass # 处理 Ctrl+D
    return "\n".join(lines)

def preprocess_text(text):
    """
    清理输入文本以适应TTS：
    1. 将换行符替换为空格。
    2. 移除除中文、英文、数字、标准中英文标点和[]之外的所有符号。
    3. 将多个空格合并为一个。
    4. 去除首尾空格。
    """
    if not isinstance(text, str):
        print("  警告：输入不是有效文本字符串。")
        return ""

    print(f"  文本预处理 - 原始文本 (预览): {repr(text[:100].strip())}...")

    # 1. 替换换行符
    cleaned_text = re.sub(r'[\r\n]+', ' ', text)

    # 2. 定义允许的字符：中文(\u4e00-\u9fff), 英文(a-zA-Z), 数字(0-9), 空格(\s),
    #    指定的中英文标点(。，、；：？！….,;:?!), 方括号([])
    #    构建正则表达式，匹配所有 *不* 在允许列表中的字符
    allowed_chars_pattern = r'[^\u4e00-\u9fff a-zA-Z0-9 。，、；：？！… \.,;:?!\\[\\]]'
    cleaned_text = re.sub(allowed_chars_pattern, '', cleaned_text)

    # 3. 合并多个空格为一个
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)

    # 4. 去除首尾空格
    cleaned_text = cleaned_text.strip()

    if not cleaned_text:
        print("  警告：文本预处理后内容为空。")
    else:
        print(f"  文本预处理 - 清理后 (预览): {repr(cleaned_text[:100])}...")
    return cleaned_text

def split_text_into_chunks(text, max_len=100):
    """
    将文本分割成适合TTS的块。优先按标点分割。
    """
    if not text:
        return []

    print(f"  准备分割文本 (长度 {len(text)}，最大块长 {max_len})")
    chunks = []
    # 使用更全面的中英文结束标点进行分割，并保留分隔符
    pattern = r'([。？！…\.?!]+)'
    sentences = re.split(pattern, text)

    # 合并句子及其结尾标点
    processed_sentences = []
    current_part = ""
    for i, part in enumerate(sentences):
        if not part: continue # 跳过空部分
        current_part += part
        # 如果 part 是分隔符或它是列表中的最后一个非空元素
        if re.fullmatch(pattern, part) or (i == len(sentences) - 1 and current_part.strip()):
            if current_part.strip():
                processed_sentences.append(current_part.strip())
            current_part = ""

    # 如果初步分割效果不佳 (比如只有一块且很长)，尝试用逗号/分号进行二级分割
    if len(processed_sentences) <= 1 and len(text) > max_len:
         print("  一级分割效果不佳，尝试使用逗号/分号进行二级分割...")
         pattern_secondary = r'([，；,;]+)'
         secondary_split = re.split(pattern_secondary, text)
         processed_sentences = [] # 重置
         current_part = ""
         for i, part in enumerate(secondary_split):
             if not part: continue
             current_part += part
             if re.fullmatch(pattern_secondary, part) or (i == len(secondary_split) - 1 and current_part.strip()):
                 if current_part.strip():
                     processed_sentences.append(current_part.strip())
                 current_part = ""

    # 合并短句以达到或接近 max_len
    current_chunk = ""
    for sentence in processed_sentences:
        if not sentence: continue

        sentence_len = len(sentence)
        current_chunk_len = len(current_chunk)

        # 如果新句子本身就超长
        if sentence_len > max_len:
            # 如果当前块有内容，先添加当前块
            if current_chunk:
                chunks.append(current_chunk)
            # 超长句子自成一块（或多块，如果需要硬切，但这里简化）
            chunks.append(sentence)
            print(f"  警告：分割出的句子 '{sentence[:30]}...' 过长 ({sentence_len} > {max_len})，可能影响合成。")
            current_chunk = "" # 重置当前块
        # 如果当前块为空，或者加上新句子不超过最大长度
        elif not current_chunk or current_chunk_len + sentence_len + 1 <= max_len: # +1 for potential space
            current_chunk += (" " if current_chunk else "") + sentence
        # 当前块加上新句子会超长
        else:
            chunks.append(current_chunk) # 添加当前块
            current_chunk = sentence # 新句子成为新的当前块

    # 添加最后一个块
    if current_chunk:
        chunks.append(current_chunk)

    # 最后检查：如果完全没有分割（例如单句且未超长），则返回包含原始文本的列表
    if not chunks and text:
        chunks = [text]
        if len(text) > max_len:
             print(f"  警告：整段文本 '{text[:30]}...' 过长 ({len(text)} > {max_len}) 且无法按标点分割，可能影响合成质量。")

    print(f"文本分割为 {len(chunks)} 个块进行处理。")
    # for i, chunk in enumerate(chunks): print(f"    块 {i+1}: {repr(chunk)}") # Debug: 打印分块结果
    return chunks


def clear_memory():
    """清理内存，释放不再使用的资源"""
    # 手动触发垃圾回收
    gc.collect()
    
    # 如果使用了CUDA，清理CUDA缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    print("内存清理完成")

def main():
    """主程序入口"""
    # --- 初始化 ---
    cosyvoice_model = load_model()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n音频输出将保存到 '{OUTPUT_DIR}' 目录。")

    file_counter = {} # 为每个 (模式+音色) 组合维护计数器

    # --- 主交互循环 ---
    while True:
        print("\n" + "="*20 + " CosyVoice 交互式语音合成 " + "="*20)

        # --- 选择音色 ---
        prompt_speech_16k, prompt_name = list_and_select_prompt_audio(ASSET_DIR)
        if prompt_speech_16k is None:
            retry = input("加载提示音频失败。是否重试? (y/n，默认为 n): ").lower()
            if retry == 'y':
                continue
            else:
                print("退出程序。")
                break

        # --- 选择模式 ---
        print("\n--- 选择合成模式 ---")
        print("1. 零样本 (Zero-shot)：从提示音频克隆音色")
        print("2. 跨语种/细粒度控制 (Cross-lingual/Fine-grained)：支持语言切换或特殊标记如 [laughter]")
        print("3. 指令/方言 (Instruct/Dialect)：通过指令改变说话方式，如“用四川话说”")
        print("0. 退出程序")

        mode_choice = input("请输入模式编号 (1, 2, 3, 或 0): ")

        # --- 根据模式获取输入 ---
        text_to_synthesize_raw = "" # 原始用户输入
        text_to_synthesize = ""     # 预处理后的文本
        prompt_text = ""            # For zero-shot
        instruction = ""            # For instruct
        mode_tag = ""               # 用于文件名和 API 调用区分
        valid_input = False         # 标志位，检查输入是否有效

        if mode_choice == '0':
            print("正在退出程序...")
            break
        elif mode_choice == '1':
            mode = "零样本"
            mode_tag = "zero_shot"
            print(f"\n--- 模式: {mode} ---")
            text_to_synthesize_raw = get_user_input("请输入要合成的主要文本:")
            prompt_text = input("请输入提示音频中的说话内容 (可选但推荐, e.g., '希望你以后能够做的比我还好呦。'): ")
            if not text_to_synthesize_raw.strip():
                print("错误：必须输入要合成的文本。")
                continue
            text_to_synthesize = preprocess_text(text_to_synthesize_raw)
            if not text_to_synthesize:
                print("错误：文本预处理后为空，无法合成。")
                continue
            valid_input = True

        elif mode_choice == '2':
            mode = "跨语种/细粒度控制"
            mode_tag = "cross_lingual" # API 通常使用 cross_lingual
            print(f"\n--- 模式: {mode} ---")
            text_to_synthesize_raw = get_user_input("请输入带控制标记的文本 (例如 '他突然[laughter]停下来' 或混合语言):")
            if not text_to_synthesize_raw.strip():
                print("错误：必须输入要合成的文本。")
                continue
            text_to_synthesize = preprocess_text(text_to_synthesize_raw)
            if not text_to_synthesize:
                print("错误：文本预处理后为空，无法合成。")
                continue
            valid_input = True

        elif mode_choice == '3':
            mode = "指令或方言"
            mode_tag = "instruct"
            print(f"\n--- 模式: {mode} ---")
            text_to_synthesize_raw = get_user_input("请输入要合成的主要文本:")
            instruction = input("请输入指令 (例如 '用四川话说这句话', '用愉快的语气说', 'speak cheerfully'): ")
            if not text_to_synthesize_raw.strip():
                print("错误：必须输入要合成的文本。")
                continue
            if not instruction.strip(): # 检查指令是否为空
                print("错误：指令模式下必须输入有效的指令。")
                continue # 返回循环开始，让用户重新选择或输入
            text_to_synthesize = preprocess_text(text_to_synthesize_raw)
            if not text_to_synthesize:
                print("错误：文本预处理后为空，无法合成。")
                continue
            valid_input = True
        else:
            print("无效的选择，请输入 1, 2, 3, 或 0。")
            continue

        # 如果输入无效（理论上不应发生，但作为保障）
        if not valid_input:
             print("未能获取有效输入，请重试。")
             continue

        # --- 文本分块 ---
        # print(f"\n准备分割文本: {repr(text_to_synthesize[:100])}...") # Debug log
        text_chunks = split_text_into_chunks(text_to_synthesize, max_len=80) # 调整块长度
        if not text_chunks:
            print("错误：输入文本为空或无法分割成有效块。")
            continue

        # --- 迭代合成每个块 ---
        all_audio_parts = []
        print("\n正在合成，请稍候...")
        synthesis_successful = True
        for i, chunk in enumerate(text_chunks):
            # 确保块内容非空
            if not chunk or chunk.isspace():
                 print(f"  跳过空块 {i+1}/{len(text_chunks)}")
                 continue

            print(f"  处理块 {i+1}/{len(text_chunks)}: {repr(chunk[:60].strip())}...")
            results = None
            try:
                # 根据模式调用对应的推理函数
                if mode_tag == "zero_shot":
                    results = cosyvoice_model.inference_zero_shot(chunk, prompt_text, prompt_speech_16k, stream=False)
                elif mode_tag == "cross_lingual":
                    results = cosyvoice_model.inference_cross_lingual(chunk, prompt_speech_16k, stream=False)
                elif mode_tag == "instruct":
                    # 优先使用 inference_instruct2 (如果存在)
                    if hasattr(cosyvoice_model, 'inference_instruct2'):
                         results = cosyvoice_model.inference_instruct2(chunk, instruction, prompt_speech_16k, stream=False)
                    elif hasattr(cosyvoice_model, 'inference_instruct'):
                         results = cosyvoice_model.inference_instruct(chunk, instruction, prompt_speech_16k, stream=False)
                    else:
                         print(f"错误：模型对象缺少 'inference_instruct' 或 'inference_instruct2' 方法。")
                         synthesis_successful = False
                         break # 无法继续合成

                # --- 结果处理 ---
                chunk_audio_found = False
                # 情况1：结果是 generator (即使 stream=False，某些实现可能仍返回 generator)
                if isinstance(results, types.GeneratorType):
                    print(f"  信息：API 返回了 generator，正在迭代处理...")
                    try:
                        for item in results:
                            # 尝试从 generator item 中提取音频
                            if isinstance(item, dict) and 'tts_speech' in item and \
                               isinstance(item['tts_speech'], torch.Tensor) and item['tts_speech'].numel() > 0:
                                print(f"    从 generator item (dict) 提取到音频 (shape: {item['tts_speech'].shape})")
                                all_audio_parts.append(item['tts_speech'].cpu())
                                chunk_audio_found = True
                                break # 假设非流式 generator 只产生一个有效结果
                            elif isinstance(item, torch.Tensor) and item.numel() > 0:
                                print(f"    从 generator item (tensor) 提取到音频 (shape: {item.shape})")
                                all_audio_parts.append(item.cpu())
                                chunk_audio_found = True
                                break # 假设非流式 generator 只产生一个有效结果
                        if not chunk_audio_found:
                             print(f"  警告：遍历 generator 完成，但未找到块 {i+1} 的有效音频数据。")
                    except Exception as gen_e:
                        print(f"  错误：处理 generator 时出错: {gen_e}")
                        traceback.print_exc()
                        # synthesis_successful = False # 可选：如果 generator 处理失败则标记失败
                # 情况2：结果是包含 'tts_speech' 的字典
                elif isinstance(results, dict) and 'tts_speech' in results and \
                   isinstance(results['tts_speech'], torch.Tensor) and results['tts_speech'].numel() > 0:
                    print(f"  从 dict 结果中提取到音频 (shape: {results['tts_speech'].shape})")
                    all_audio_parts.append(results['tts_speech'].cpu())
                    chunk_audio_found = True
                # 情况3：结果直接是 Tensor
                elif isinstance(results, torch.Tensor) and results.numel() > 0:
                     print(f"  提取到 Tensor 结果音频 (shape: {results.shape})")
                     all_audio_parts.append(results.cpu())
                     chunk_audio_found = True
                # 情况4：结果是列表 (可能包含多个 segment)
                elif isinstance(results, list):
                    print(f"  处理 list 结果...")
                    for segment in results:
                         if isinstance(segment, dict) and 'tts_speech' in segment and \
                            isinstance(segment['tts_speech'], torch.Tensor) and segment['tts_speech'].numel() > 0:
                             print(f"    从 list item (dict) 提取到音频 (shape: {segment['tts_speech'].shape})")
                             all_audio_parts.append(segment['tts_speech'].cpu())
                             chunk_audio_found = True # 只要列表里有一个有效就认为找到了
                         elif isinstance(segment, torch.Tensor) and segment.numel() > 0:
                             print(f"    从 list item (tensor) 提取到音频 (shape: {segment.shape})")
                             all_audio_parts.append(segment.cpu())
                             chunk_audio_found = True
                # --- 结束结果处理 ---

                # 如果经过各种检查后，当前块仍未找到音频
                if not chunk_audio_found:
                     print(f"警告：块 {i+1} 未返回有效的音频数据或未能从返回结果中提取。返回结果类型: {type(results)}")

            except NotImplementedError as nie:
                 print(f"\n警告：模型不支持块 {i+1} 的所选操作模式或参数组合: {nie}")
                 print("尝试继续处理后续文本块...")
                 # 不中断整个合成过程
            except Exception as e:
                print(f"\n警告：处理块 {i+1} ('{chunk[:30]}...') 时发生错误: {e}")
                traceback.print_exc() # 打印详细错误信息
                print("尝试继续处理后续文本块...")
                # 不设置 synthesis_successful = False，也不使用 break
                # 这样即使一个块失败，也会继续处理后续块

        # --- 合并并保存 ---
        if synthesis_successful and all_audio_parts:
            try:
                print(f"\n准备合并 {len(all_audio_parts)} 个音频片段...")
                # 确保所有片段都在 CPU 上
                all_audio_parts_cpu = [part.cpu() for part in all_audio_parts]
                if len(all_audio_parts_cpu) > 1:
                    final_audio = torch.cat(all_audio_parts_cpu, dim=-1) # 沿着时间轴拼接
                elif len(all_audio_parts_cpu) == 1:
                    final_audio = all_audio_parts_cpu[0]
                else:
                    # 理论上 synthesis_successful=True 且 all_audio_parts 非空，这里不应到达
                    print("\n错误：合成声称成功但音频部分列表为空（合并前检查）。")
                    continue

                # --- 文件名生成 ---
                counter_key = (mode_tag, prompt_name)
                current_count = file_counter.get(counter_key, 0) + 1
                file_counter[counter_key] = current_count

                # 使用原始文本生成前缀，限制长度并清理
                text_prefix_raw = text_to_synthesize_raw[:20].strip() # 增加前缀长度
                safe_text_prefix = re.sub(r'[^\w\u4e00-\u9fff\- ]', '_', text_prefix_raw).strip().replace(' ', '_')
                safe_text_prefix = safe_text_prefix[:30] # 限制清理后的前缀长度

                # 清理音色名
                safe_prompt_name = re.sub(r'[^\w\u4e00-\u9fff\-]', '_', prompt_name).strip()

                # 添加模式和指令信息
                extra_info = ""
                if mode_tag == "cross_lingual":
                    if re.search(r'\[\w+\]', text_to_synthesize): extra_info += "_标记"
                    # 基础语言判断
                    has_en = re.search(r'[a-zA-Z]', text_to_synthesize)
                    has_zh = re.search(r'[\u4e00-\u9fff]', text_to_synthesize)
                    if has_en and has_zh: extra_info += "_混合"
                    elif has_en: extra_info += "_英文"
                elif mode_tag == "instruct":
                    # 清理并缩短指令作为文件名一部分
                    instr_part = re.sub(r'[^\w\u4e00-\u9fff\-]', '_', instruction[:15]).strip()
                    if instr_part: extra_info = f"_{instr_part}"

                if not safe_text_prefix: safe_text_prefix = "合成结果" # 默认前缀

                # 构建基本文件名并限制总长度
                output_filename_base = f"输出_{safe_text_prefix}_{mode_tag}_{safe_prompt_name}{extra_info}_{current_count}"
                max_len_base = 180 # 进一步缩短以适应不同文件系统
                if len(output_filename_base) > max_len_base:
                    output_filename_base = output_filename_base[:max_len_base] + "_etc"

                output_filename = os.path.join(OUTPUT_DIR, f"{output_filename_base}.wav")
                # --- 文件名生成结束 ---

                # 获取模型采样率，提供默认值
                sample_rate = int(getattr(cosyvoice_model, 'sample_rate', 22050))
                print(f"准备以 {sample_rate} Hz 保存音频...")
                torchaudio.save(output_filename, final_audio, sample_rate)
                print("\n✅ 合成完成！音频已保存至:")
                print(f"   - {output_filename}")
                
                # 合成完成后清理内存
                all_audio_parts.clear()  # 清空音频片段列表
                all_audio_parts_cpu.clear()  # 清空 CPU 版音频片段
                del final_audio  # 删除最终音频引用
                clear_memory()  # 调用内存清理函数

            except Exception as e:
                print(f"\n错误：合并或保存最终音频文件时出错: {e}")
                traceback.print_exc()
                # 即使出错也尝试清理内存
                clear_memory()
        elif synthesis_successful and not all_audio_parts:
             print("\n⚠️ 合成过程似乎已完成，但未能生成任何有效的音频片段。请检查输入文本、指令和模型日志。")
        else: # synthesis_successful is False
             print("\n❌ 由于处理块时出错，合成未完成。")


        # 在每次合成结束后清理内存，无论成功与否
        if not all_audio_parts:
            clear_memory()
            
        print("-" * 40)
        # 询问是否继续
        try:
            cont = input("是否进行下一次合成? (y/n，默认为 y): ").lower().strip()
            if cont == 'n':
                print("用户选择退出。")
                break
            # 每次循环结束时强制清理内存
            clear_memory()
        except EOFError:
            print("\n检测到输入结束，退出程序。")
            break # 退出循环
        # 循环继续，进行下一次合成

if __name__ == "__main__":
    # 可以在这里设置全局编码，但通常不是必须的
    # try:
    #     sys.stdout.reconfigure(encoding='utf-8')
    #     sys.stdin.reconfigure(encoding='utf-8')
    # except AttributeError:
    #     print("Info: sys.stdout/stdin.reconfigure not available on this Python version/OS.")
    #     pass # 在不支持的环境中跳过

    main()
    print("\n程序执行完毕。")