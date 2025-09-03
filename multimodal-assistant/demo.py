#!/usr/bin/env python3
"""
🎮 Interactive Demo Script for Multimodal AI Assistant

Hướng dẫn test các tính năng của hệ thống một cách interactive.
"""

import asyncio
import aiohttp
import json
import os
import tempfile
import wave
import struct
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import sys

BASE_URL = "http://localhost:8000"

class Colors:
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.PURPLE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text.center(60)}{Colors.END}")
    print(f"{Colors.PURPLE}{'='*60}{Colors.END}\n")

def print_success(text):
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")

def print_error(text):
    print(f"{Colors.RED}❌ {text}{Colors.END}")

def print_info(text):
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.END}")

def print_warning(text):
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.END}")

async def check_server():
    """Kiểm tra server có chạy không"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{BASE_URL}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    print_success(f"Server đang chạy - Version: {data.get('version', 'unknown')}")
                    return True
                else:
                    print_error(f"Server trả về status code: {response.status}")
                    return False
    except aiohttp.ClientConnectorError:
        print_error("Không thể kết nối đến server!")
        print_info("Hãy chạy: make serve hoặc docker-compose up")
        return False
    except Exception as e:
        print_error(f"Lỗi không xác định: {e}")
        return False

def create_sample_image():
    """Tạo ảnh mẫu để test OCR"""
    img = Image.new('RGB', (400, 200), color='white')
    draw = ImageDraw.Draw(img)
    
    # Vẽ text
    text = "Multimodal AI Assistant\nDemo Test Image\nOCR Testing 2024"
    draw.text((50, 50), text, fill='black')
    
    # Vẽ hình chữ nhật
    draw.rectangle([300, 50, 350, 100], outline='blue', width=2)
    
    # Lưu file tạm
    temp_path = tempfile.mktemp(suffix='.png')
    img.save(temp_path)
    return temp_path

def create_sample_audio():
    """Tạo file audio mẫu (sine wave)"""
    duration = 2.0  # 2 seconds
    sample_rate = 16000
    frequency = 440  # A4 note
    
    # Generate sine wave
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave_data = np.sin(frequency * 2 * np.pi * t)
    
    # Convert to 16-bit integers
    wave_data = (wave_data * 32767).astype(np.int16)
    
    # Save to temp file
    temp_path = tempfile.mktemp(suffix='.wav')
    with wave.open(temp_path, 'wb') as wav_file:
        wav_file.setnchannels(1)  # mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(wave_data.tobytes())
    
    return temp_path

async def demo_health_check():
    """Demo health check endpoint"""
    print_header("🏥 HEALTH CHECK DEMO")
    
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{BASE_URL}/health") as response:
            data = await response.json()
            print_info(f"GET {BASE_URL}/health")
            print(f"Response: {json.dumps(data, indent=2)}")
            print_success("Health check thành công!")

async def demo_multimodal_analysis():
    """Demo multimodal analysis"""
    print_header("🤖 MULTIMODAL ANALYSIS DEMO")
    
    # Tạo ảnh mẫu
    print_info("Tạo ảnh mẫu cho demo...")
    image_path = create_sample_image()
    
    try:
        async with aiohttp.ClientSession() as session:
            # Test OCR
            print_info("Testing OCR...")
            data = aiohttp.FormData()
            data.add_field('tasks', '["ocr"]')
            data.add_field('image', open(image_path, 'rb'), filename='demo.png')
            
            async with session.post(f"{BASE_URL}/v1/multimodal/analyze", data=data) as response:
                result = await response.json()
                print(f"OCR Result: {json.dumps(result, indent=2)}")
                
                if result.get('ok') and 'ocr' in result.get('result', {}):
                    print_success(f"OCR extracted: '{result['result']['ocr']}'")
                else:
                    print_warning("OCR không trích xuất được text")
            
            # Test Text Summarization
            print_info("\nTesting Text Summarization...")
            data = aiohttp.FormData()
            data.add_field('text', 'Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of "intelligent agents": any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals.')
            data.add_field('tasks', '["summary"]')
            
            async with session.post(f"{BASE_URL}/v1/multimodal/analyze", data=data) as response:
                result = await response.json()
                print(f"Summary Result: {json.dumps(result, indent=2)}")
                
                if result.get('ok') and 'summary' in result.get('result', {}):
                    print_success(f"Summary: '{result['result']['summary']}'")
                else:
                    print_warning("Summarization không thành công")
                    
    finally:
        # Cleanup
        if os.path.exists(image_path):
            os.remove(image_path)

async def demo_rag():
    """Demo RAG functionality"""
    print_header("📚 RAG (RETRIEVAL-AUGMENTED GENERATION) DEMO")
    
    # Upload sample document
    print_info("Uploading sample document...")
    
    sample_text = """
    Multimodal AI Assistant là một hệ thống AI tiên tiến có khả năng xử lý nhiều loại dữ liệu khác nhau.
    Hệ thống này có thể:
    1. Xử lý văn bản: tóm tắt, trả lời câu hỏi
    2. Xử lý hình ảnh: OCR, VQA  
    3. Xử lý âm thanh: ASR, TTS, VAD
    4. Tích hợp RAG để tìm kiếm thông tin
    
    Công nghệ sử dụng bao gồm PyTorch, Transformers, FastAPI và Docker.
    """
    
    temp_text_file = tempfile.mktemp(suffix='.txt')
    with open(temp_text_file, 'w', encoding='utf-8') as f:
        f.write(sample_text)
    
    try:
        async with aiohttp.ClientSession() as session:
            # Upload document
            data = aiohttp.FormData()
            data.add_field('files', open(temp_text_file, 'rb'), filename='demo.txt')
            
            async with session.post(f"{BASE_URL}/v1/rag/upload", data=data) as response:
                result = await response.json()
                print(f"Upload Result: {json.dumps(result, indent=2)}")
                
                if result.get('ok'):
                    print_success(f"Uploaded thành công! Chunks added: {result.get('chunks_added', 0)}")
                else:
                    print_error("Upload thất bại")
                    return
            
            # Query RAG
            print_info("\nQuerying RAG...")
            data = aiohttp.FormData()
            data.add_field('question', 'Hệ thống có những khả năng gì?')
            data.add_field('top_k', '3')
            
            async with session.post(f"{BASE_URL}/v1/rag/query", data=data) as response:
                result = await response.json()
                print(f"Query Result: {json.dumps(result, indent=2)}")
                
                if result.get('ok'):
                    answer = result.get('result', {}).get('answer', {})
                    print_success(f"RAG Answer: '{answer.get('answer', '')}'")
                else:
                    print_error("Query thất bại")
                    
    finally:
        # Cleanup
        if os.path.exists(temp_text_file):
            os.remove(temp_text_file)

async def demo_audio():
    """Demo audio processing"""
    print_header("🎵 AUDIO PROCESSING DEMO")
    
    # Create sample audio
    print_info("Tạo file audio mẫu...")
    audio_path = create_sample_audio()
    
    try:
        async with aiohttp.ClientSession() as session:
            # Test TTS first
            print_info("Testing Text-to-Speech...")
            data = aiohttp.FormData()
            data.add_field('text', 'Hello from multimodal AI assistant!')
            
            async with session.post(f"{BASE_URL}/v1/audio/tts", data=data) as response:
                if response.status == 200:
                    audio_data = await response.read()
                    tts_path = tempfile.mktemp(suffix='.wav')
                    with open(tts_path, 'wb') as f:
                        f.write(audio_data)
                    print_success(f"TTS thành công! Audio saved to: {tts_path}")
                else:
                    print_warning("TTS không thành công")
            
            # Test VAD
            print_info("\nTesting Voice Activity Detection...")
            data = aiohttp.FormData()
            data.add_field('audio', open(audio_path, 'rb'), filename='demo.wav')
            
            async with session.post(f"{BASE_URL}/v1/audio/vad", data=data) as response:
                result = await response.json()
                print(f"VAD Result: {json.dumps(result, indent=2)}")
                
                if result.get('ok'):
                    segments = result.get('segments', [])
                    print_success(f"VAD detected {len(segments)} speech segments")
                else:
                    print_warning("VAD không thành công")
            
            # Test ASR
            print_info("\nTesting Automatic Speech Recognition...")
            data = aiohttp.FormData()
            data.add_field('audio', open(audio_path, 'rb'), filename='demo.wav')
            
            async with session.post(f"{BASE_URL}/v1/audio/asr", data=data) as response:
                result = await response.json()
                print(f"ASR Result: {json.dumps(result, indent=2)}")
                
                if result.get('ok'):
                    text = result.get('text', '')
                    print_success(f"ASR transcription: '{text}'")
                else:
                    print_warning("ASR không thành công")
                    
    finally:
        # Cleanup
        if os.path.exists(audio_path):
            os.remove(audio_path)

async def demo_xai():
    """Demo Explainable AI"""
    print_header("🔍 EXPLAINABLE AI (XAI) DEMO")
    
    # Create sample image
    print_info("Tạo ảnh mẫu cho Grad-CAM...")
    image_path = create_sample_image()
    
    try:
        async with aiohttp.ClientSession() as session:
            data = aiohttp.FormData()
            data.add_field('image', open(image_path, 'rb'), filename='demo.png')
            
            async with session.post(f"{BASE_URL}/v1/xai/gradcam", data=data) as response:
                if response.status == 200:
                    gradcam_data = await response.read()
                    gradcam_path = tempfile.mktemp(suffix='.png')
                    with open(gradcam_path, 'wb') as f:
                        f.write(gradcam_data)
                    print_success(f"Grad-CAM thành công! Saved to: {gradcam_path}")
                    print_info(f"Bạn có thể mở file để xem: {gradcam_path}")
                else:
                    result = await response.json()
                    print_error(f"Grad-CAM thất bại: {result}")
                    
    finally:
        # Cleanup
        if os.path.exists(image_path):
            os.remove(image_path)

async def run_full_demo():
    """Chạy full demo tất cả tính năng"""
    print_header("🚀 MULTIMODAL AI ASSISTANT - FULL DEMO")
    
    # Check server first
    if not await check_server():
        return
    
    try:
        await demo_health_check()
        await demo_multimodal_analysis()
        await demo_rag()
        await demo_audio()
        await demo_xai()
        
        print_header("🎉 DEMO HOÀN THÀNH")
        print_success("Tất cả tính năng đã được test thành công!")
        print_info("Hãy explore API thêm tại: http://localhost:8000/docs")
        
    except KeyboardInterrupt:
        print_warning("\nDemo bị interrupt bởi user")
    except Exception as e:
        print_error(f"Lỗi trong quá trình demo: {e}")

async def interactive_menu():
    """Menu interactive cho user lựa chọn"""
    while True:
        print_header("🎮 INTERACTIVE DEMO MENU")
        print("1. 🏥 Health Check")
        print("2. 🤖 Multimodal Analysis (OCR, Summarization)")
        print("3. 📚 RAG Demo")
        print("4. 🎵 Audio Processing")
        print("5. 🔍 Explainable AI (Grad-CAM)")
        print("6. 🚀 Run Full Demo")
        print("7. ❌ Exit")
        
        choice = input(f"\n{Colors.CYAN}Chọn option (1-7): {Colors.END}").strip()
        
        if choice == '1':
            await demo_health_check()
        elif choice == '2':
            await demo_multimodal_analysis()
        elif choice == '3':
            await demo_rag()
        elif choice == '4':
            await demo_audio()
        elif choice == '5':
            await demo_xai()
        elif choice == '6':
            await run_full_demo()
        elif choice == '7':
            print_success("Goodbye! 👋")
            break
        else:
            print_warning("Invalid choice. Please select 1-7.")
        
        input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.END}")

async def main():
    """Main function"""
    if len(sys.argv) > 1 and sys.argv[1] == '--full':
        await run_full_demo()
    else:
        await interactive_menu()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print_warning("\nDemo interrupted. Goodbye! 👋")
    except Exception as e:
        print_error(f"Unexpected error: {e}")
