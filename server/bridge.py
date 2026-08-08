#!/usr/bin/env python3
"""WebSocket Bridge Server for StackChan
Receives audio from device, transcribes it, sends to AI, returns TTS audio.
"""
import asyncio
import json
import struct
import os
import aiohttp
from aiohttp import web
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('bridge')

# Configuration
API_URL = os.getenv('API_URL', 'https://your-api-endpoint.com/v1/chat/completions')
API_KEY = os.getenv('API_KEY', 'sk-your-key')
TTS_URL = os.getenv('TTS_URL', 'https://your-tts-endpoint.com/v1/audio/speech')
STT_URL = os.getenv('STT_URL', 'https://your-stt-endpoint.com/v1/audio/transcriptions')
SYSTEM_PROMPT = os.getenv('SYSTEM_PROMPT', 'You are a helpful assistant.')
SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000

class DeviceSession:
    def __init__(self, ws):
        self.ws = ws
        self.session_id = None
        self.audio_buffer = bytearray()
        self.is_listening = False
        self.conversation_history = []
    
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
        self.last_audio_time = asyncio.get_event_loop().time()
        
        # Start a timer to process after silence
        if not hasattr(self, '_audio_task') or self._audio_task is None or self._audio_task.done():
            self._audio_task = asyncio.create_task(self._wait_and_process())
    
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
                # Wake word detected, start listening
                self.audio_buffer = bytearray()
                self.is_listening = True
                logger.info("Wake word detected, listening started")
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
                        'model': 'claude-sonnet-4-20250514',
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
    
    async def send_tts_message(self, text):
        """Send text response to device for TTS playback"""
        msg = {
            'type': 'tts',
            'state': 'start',
            'text': text
        }
        await self.ws.send_str(json.dumps(msg))
        
        # Signal end of TTS
        end_msg = {
            'type': 'tts',
            'state': 'stop'
        }
        await self.ws.send_str(json.dumps(end_msg))


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


def create_app():
    app = web.Application()
    app.router.add_get('/xiaozhi/v1/', websocket_handler)
    app.router.add_get('/xiaozhi/ota/', ota_handler)
    app.router.add_post('/xiaozhi/ota/', ota_handler)
    return app


if __name__ == '__main__':
    app = create_app()
    port = int(os.getenv('PORT', '8003'))
    logger.info(f"Starting bridge server on port {port}")
    web.run_app(app, host='0.0.0.0', port=port)
