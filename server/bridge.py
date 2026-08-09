#!/usr/bin/env python3
"""WebSocket Bridge Server for StackChan
Receives audio from device, transcribes it, sends to AI, returns TTS audio.
"""
import asyncio
import json
import os
import time
import aiohttp
from aiohttp import web
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('bridge')

# Configuration
API_URL = os.getenv('API_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions')
API_KEY = os.getenv('API_KEY', 'sk-your-key')
STT_URL = os.getenv('STT_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1/audio/transcriptions')
SYSTEM_PROMPT = os.getenv('SYSTEM_PROMPT', 'You are a helpful assistant.')
MODEL = os.getenv('MODEL', 'qwen-plus')
SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000

class DeviceSession:
    def __init__(self, ws):
        self.ws = ws
        self.session_id = None
        self.audio_buffer = bytearray()
        self.is_listening = False
        self.conversation_history = []
        self.last_audio_time = 0
        self._silence_task = None
    
    async def handle_hello(self, msg):
        """Handle hello handshake from device"""
        logger.info(f"Device hello received: {msg}")
        self.session_id = 'session_' + str(id(self))
        
        response = {
            'type': 'hello',
            'transport': 'websocket',
            'session_id': self.session_id,
            'audio_params': {
                'format': 'opus',
                'sample_rate': OUTPUT_SAMPLE_RATE,
                'channels': 1,
                'frame_duration': 60
            }
        }
        await self.ws.send_str(json.dumps(response))
        logger.info(f"Hello response sent, session: {self.session_id}")
    
    async def handle_audio(self, data):
        """Buffer incoming audio data and auto-process after silence"""
        self.audio_buffer.extend(data)
        self.last_audio_time = time.time()
        
        # Cancel previous silence timer and start a new one
        if self._silence_task and not self._silence_task.done():
            self._silence_task.cancel()
        self._silence_task = asyncio.create_task(self._check_silence())
    
    async def handle_text_message(self, msg):
        """Handle JSON text messages from device"""
        msg_type = msg.get('type', '')
        
        if msg_type == 'hello':
            await self.handle_hello(msg)
        elif msg_type == 'listen':
            state = msg.get('state', '')
            text = msg.get('text', '')
            logger.info(f"Listen event: state={state}, text={text}")
            if state == 'detect':
                # Debounce: ignore repeated detect within 3 seconds
                now = time.time()
                if hasattr(self, '_last_detect_time') and now - self._last_detect_time < 3:
                    logger.info(f"Debounce: skipping repeated detect")
                    return
                self._last_detect_time = now
                
                self.audio_buffer = bytearray()
                self.is_listening = True
                self.last_audio_time = now
                logger.info(f"Wake word detected: {text}")
                # Respond to the touch interaction
                if text:
                    asyncio.create_task(self.respond_to_input(text))
            elif state == 'start':
                self.audio_buffer = bytearray()
                self.is_listening = True
                logger.info("Listening started")
            elif state == 'stop':
                self.is_listening = False
                logger.info(f"Listening stopped, buffer size: {len(self.audio_buffer)}")
                await self.process_audio()
        elif msg_type == 'stt':
            state = msg.get('state', '')
            if state == 'start':
                self.audio_buffer = bytearray()
                self.is_listening = True
            elif state == 'stop':
                self.is_listening = False
                await self.process_audio()
        elif msg_type == 'abort':
            logger.info("Abort received")
    
    async def process_audio(self):
        """Transcribe audio and get AI response"""
        if not self.audio_buffer:
            return
        
        # Step 1: STT - transcribe audio
        transcript = await self.transcribe(self.audio_buffer)
        if not transcript:
            await self.send_tts_message("Sorry, I didn't catch that.")
            return
        
        logger.info(f"Transcribed: {transcript}")
        
        # Step 2: Send to AI
        self.conversation_history.append({'role': 'user', 'content': transcript})
        response = await self.get_ai_response(transcript)
        if response:
            self.conversation_history.append({'role': 'assistant', 'content': response})
            logger.info(f"AI response: {response}")
            # Step 3: Send response back as text (device will use its own TTS)
            await self.send_tts_message(response)
    
    async def transcribe(self, audio_data):
        """Send audio to STT service"""
        try:
            async with aiohttp.ClientSession() as session:
                form = aiohttp.FormData()
                form.add_field('file', bytes(audio_data), filename='audio.opus', content_type='audio/opus')
                form.add_field('model', 'whisper-1')
                form.add_field('language', 'zh')
                
                async with session.post(
                    STT_URL,
                    data=form,
                    headers={'Authorization': f'Bearer {API_KEY}'}
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        return result.get('text', '')
                    else:
                        logger.error(f"STT failed: {resp.status}")
                        return None
        except Exception as e:
            logger.error(f"STT error: {e}")
            return None
    
    async def get_ai_response(self, text):
        """Get response from AI model"""
        try:
            messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]
            # Keep last 20 messages for context
            messages.extend(self.conversation_history[-20:])
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    API_URL,
                    json={
                        'model': os.getenv('MODEL', 'qwen-plus'),
                        'messages': messages,
                        'max_tokens': 500
                    },
                    headers={
                        'Authorization': f'Bearer {API_KEY}',
                        'Content-Type': 'application/json'
                    }
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        return result['choices'][0]['message']['content']
                    else:
                        logger.error(f"AI API failed: {resp.status}")
                        return None
        except Exception as e:
            logger.error(f"AI error: {e}")
            return None
    
    async def respond_to_input(self, text):
        """Respond to touch or voice input"""
        try:
            self._responded_via_detect = True
            self.conversation_history.append({'role': 'user', 'content': text})
            response = await self.get_ai_response(text)
            if not response:
                response = '嗯~'
            self.conversation_history.append({'role': 'assistant', 'content': response})
            logger.info(f"AI response: {response}")
            await self.send_tts_message(response)
        except Exception as e:
            logger.error(f"respond_to_input error: {e}")

    async def _check_silence(self):
        """Wait for silence then process audio"""
        await asyncio.sleep(2.0)
        # Skip if already responded via touch detect
        if hasattr(self, '_responded_via_detect') and self._responded_via_detect:
            self._responded_via_detect = False
            self.audio_buffer = bytearray()
            return
        # If no new audio in the last 1.8 seconds, consider speech ended
        if time.time() - self.last_audio_time >= 1.8 and self.audio_buffer:
            logger.info(f"Silence detected, processing {len(self.audio_buffer)} bytes of audio")
            await self.process_audio()

    async def process_audio(self):
        """Transcribe audio and get AI response"""
        if not self.audio_buffer:
            return
        
        audio_data = bytes(self.audio_buffer)
        self.audio_buffer = bytearray()
        self.is_listening = False
        
        logger.info(f"Processing {len(audio_data)} bytes of audio")
        
        # Step 1: STT - transcribe audio
        transcript = await self.transcribe(audio_data)
        if not transcript:
            logger.warning("STT failed, using fallback")
            transcript = '你好'
        
        logger.info(f"Transcribed: {transcript}")
        
        # Send STT result to device
        await self.ws.send_str(json.dumps({
            'type': 'stt',
            'text': transcript
        }))
        
        # Step 2: Get AI response
        self.conversation_history.append({'role': 'user', 'content': transcript})
        response = await self.get_ai_response(transcript)
        if not response:
            response = '你好呀'
        self.conversation_history.append({'role': 'assistant', 'content': response})
        logger.info(f"AI response: {response}")
        
        # Step 3: Send TTS response
        await self.send_tts_message(response)

    async def send_tts_message(self, text):
        """Generate TTS audio and send to device as raw opus frames"""
        import tempfile
        import struct
        import opuslib
        import wave
        
        try:
            # Step 1: Generate audio with edge-tts
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
                mp3_path = f.name
            
            edge_tts_bin = '/root/.espressif/python_env/idf5.5_py3.11_env/bin/edge-tts'
            # Strip markdown/action text for cleaner speech
            clean_text = text.replace('*', '').strip()
            if not clean_text:
                clean_text = '嗯'
            # Limit text length
            clean_text = clean_text[:200]
            
            proc = await asyncio.create_subprocess_exec(
                edge_tts_bin, '--text', clean_text, '--voice', 'zh-CN-YunxiNeural',
                '--write-media', mp3_path,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            await proc.wait()
            
            if not os.path.exists(mp3_path) or os.path.getsize(mp3_path) == 0:
                logger.error("TTS generation failed")
                return
            
            # Step 2: Convert mp3 to raw PCM wav (24kHz mono 16-bit)
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                wav_path = f.name
            
            proc = await asyncio.create_subprocess_exec(
                'ffmpeg', '-y', '-i', mp3_path,
                '-ar', '24000', '-ac', '1', '-f', 'wav',
                wav_path,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            await proc.wait()
            
            # Step 3: Read PCM data from wav
            with wave.open(wav_path, 'rb') as wf:
                pcm_data = wf.readframes(wf.getnframes())
            
            # Step 4: Encode PCM to opus frames (60ms per frame = 1440 samples at 24kHz)
            encoder = opuslib.Encoder(24000, 1, opuslib.APPLICATION_AUDIO)
            frame_size = 1440  # 60ms at 24kHz
            frame_bytes = frame_size * 2  # 16-bit = 2 bytes per sample
            
            # Send TTS start
            await self.ws.send_str(json.dumps({
                'type': 'tts',
                'state': 'start'
            }))
            
            total_sent = 0
            for i in range(0, len(pcm_data), frame_bytes):
                frame = pcm_data[i:i+frame_bytes]
                # Pad last frame if needed
                if len(frame) < frame_bytes:
                    frame = frame + b'\x00' * (frame_bytes - len(frame))
                
                # Encode to opus
                opus_frame = encoder.encode(frame, frame_size)
                await self.ws.send_bytes(opus_frame)
                total_sent += len(opus_frame)
                await asyncio.sleep(0.02)  # Increased delay for device decode timing
            
            # Send TTS stop
            await self.ws.send_str(json.dumps({
                'type': 'tts',
                'state': 'stop'
            }))
            
            logger.info(f"TTS sent: {total_sent} bytes ({total_sent // frame_bytes} frames)")
            
            # Cleanup
            os.unlink(mp3_path)
            os.unlink(wav_path)
            
        except Exception as e:
            logger.error(f"TTS error: {e}")


async def websocket_handler(request):
    """Handle WebSocket connections from StackChan device"""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    session = DeviceSession(ws)
    logger.info("Device connected")
    
    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.BINARY:
                await session.handle_audio(msg.data)
            elif msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    await session.handle_text_message(data)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON: {msg.data}")
            elif msg.type == aiohttp.WSMsgType.ERROR:
                logger.error(f"WebSocket error: {ws.exception()}")
    except Exception as e:
        logger.error(f"Session error: {e}")
    finally:
        logger.info("Device disconnected")
    
    return ws


async def ota_handler(request):
    """Handle OTA check - return websocket URL and version info"""
    server_ip = os.getenv('SERVER_IP', '101.200.241.96')
    port = os.getenv('PORT', '8003')
    response = {
        'server_time': {
            'timestamp': int(asyncio.get_event_loop().time()),
            'timezone_offset': 480
        },
        'firmware': {
            'version': '2.2.6',
            'url': ''
        },
        'websocket': {
            'url': f'ws://{server_ip}:{port}/xiaozhi/v1/',
            'token': ''
        }
    }
    return web.json_response(response)


async def send_email_handler(request):
    """Send email via 163 SMTP"""
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    
    try:
        data = await request.json()
        to_addr = data.get('to', '')
        subject = data.get('subject', '')
        body = data.get('body', '')
        
        if not to_addr or not body:
            return web.json_response({'error': 'Missing to or body'}, status=400)
        
        msg = MIMEText(body, 'plain', 'utf-8')
        msg['Subject'] = subject
        msg['From'] = 'evanluchen26@163.com'
        msg['To'] = to_addr
        
        with smtplib.SMTP_SSL('smtp.163.com', 465) as s:
            s.login('evanluchen26@163.com', 'TJ3kzAZDgyKrszfn')
            s.send_message(msg)
        
        return web.json_response({'status': 'sent', 'to': to_addr, 'subject': subject})
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)


@web.middleware
async def cors_middleware(request, handler):
    if request.method == 'OPTIONS':
        response = web.Response()
    else:
        response = await handler(request)
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response


def create_app():
    app = web.Application(middlewares=[cors_middleware])
    app.router.add_get('/xiaozhi/v1/', websocket_handler)
    app.router.add_get('/xiaozhi/ota/', ota_handler)
    app.router.add_post('/xiaozhi/ota/', ota_handler)
    app.router.add_post('/send-email', send_email_handler)
    return app


if __name__ == '__main__':
    app = create_app()
    port = int(os.getenv('PORT', '8003'))
    logger.info(f"Starting bridge server on port {port}")
    web.run_app(app, host='0.0.0.0', port=port)
