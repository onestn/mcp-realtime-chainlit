import chainlit as cl
from uuid import uuid4
from chainlit.logger import logger

from realtime import RealtimeClient

from dotenv import load_dotenv
load_dotenv(override=True)

# Load translations configuration
try:
    import chainlit_config  # noqa: F401 - Import needed for side effects
    logger.info("Loaded chainlit_config for translations")
except Exception as e:
    logger.warning(f"Failed to load chainlit_config: {e}")

async def setup_openai_realtime():
    """Instantiate and configure the OpenAI Realtime Client"""
             
    openai_realtime = RealtimeClient(system_prompt = "")
    cl.user_session.set("track_id", str(uuid4()))
    # Initialize the flag to track input type (text vs audio)
    cl.user_session.set("is_text_input", True)
    async def handle_conversation_updated(event):
        item = event.get("item")
        delta = event.get("delta")
        """Currently used to stream audio back to the client."""
        if event:
            # print(f"Event {event}")
            # Skip handling input_audio_transcription events when text is typed (not audio)
            # This prevents duplicate messages for non-Latin scripts
            if "input_audio_transcription" in item["type"] and not cl.user_session.get("is_text_input", False):
                msg = cl.Message(content=delta["transcript"], author="user")
                msg.type = "user_message"
                await msg.send()
        if delta:
            # Only one of the following will be populated for any given event
            if 'audio' in delta:
                audio = delta['audio']  # Int16Array, audio added
                await cl.context.emitter.send_audio_chunk(cl.OutputAudioChunk(mimeType="pcm16", data=audio, track=cl.user_session.get("track_id")))
            if 'transcript' in delta:
                transcript = delta['transcript']  # string, transcript added
                # Display realtime audio transcription in chat
                if item and item.get("role") == "assistant":
                    # Get or create message for streaming transcript
                    transcript_msg = cl.user_session.get("current_transcript_msg")
                    if not transcript_msg:
                        transcript_msg = cl.Message(content="", author="assistant")
                        cl.user_session.set("current_transcript_msg", transcript_msg)
                        await transcript_msg.send()
                    
                    # Update transcript content
                    transcript_msg.content += transcript
                    await transcript_msg.update()
            if 'arguments' in delta:
                # Arguments added but not used in this context
                pass
            
    async def handle_item_completed(item):
        """Used to populate the chat context with transcription once an item is completed."""
        # print(f"Item {item}")
        if item["item"]["type"] == "message":
            content = item["item"]["content"][0]
            # print(f"Content {content}")
            if content["type"] == "audio":
                # Clear the streaming transcript message since the item is completed
                cl.user_session.set("current_transcript_msg", None)
                # Only send completed message if it was audio input, not text input
                if not cl.user_session.get("is_text_input", False):
                    await cl.Message(content=content["transcript"], author="assistant").send()
        elif item["item"]["type"] == "function_call":
            # Clear transcript message for function calls too
            cl.user_session.set("current_transcript_msg", None)
    
    async def handle_conversation_interrupt(event):
        """Used to cancel the client previous audio playback."""
        cl.user_session.set("track_id", str(uuid4()))
        # NOTE this will only work starting from version 2.0.0
        await cl.context.emitter.send_audio_interrupt()
        
    async def handle_error(event):
        logger.error(event)
        
    
    openai_realtime.on('conversation.updated', handle_conversation_updated)
    openai_realtime.on('conversation.item.completed', handle_item_completed)
    openai_realtime.on('conversation.interrupted', handle_conversation_interrupt)
    openai_realtime.on('error', handle_error)

    cl.user_session.set("openai_realtime", openai_realtime)
    

@cl.on_chat_start
async def start():
    await setup_openai_realtime()
    try:
        openai_realtime: RealtimeClient = cl.user_session.get("openai_realtime")
        await openai_realtime.connect()
        logger.info("Connected to OpenAI realtime")
        return True
    except Exception as e:
        await cl.ErrorMessage(content=f"Failed to connect to OpenAI realtime: {e}").send()
        return False


@cl.on_message
async def on_message(message: cl.Message):
    openai_realtime: RealtimeClient = cl.user_session.get("openai_realtime")
    if openai_realtime and openai_realtime.is_connected():
        # Set flag to indicate this is text input, not audio
        cl.user_session.set("is_text_input", True)
        await openai_realtime.send_user_message_content([{ "type": 'input_text', "text": message.content}])
    else:
        await cl.Message(content="Please activate voice mode before sending messages!").send()

@cl.on_audio_start
async def on_audio_start():
    return True
    # try:
    #     openai_realtime: RealtimeClient = cl.user_session.get("openai_realtime")
    #     await openai_realtime.connect()
    #     logger.info("Connected to OpenAI realtime")
    #     return True
    # except Exception as e:
    #     await cl.ErrorMessage(content=f"Failed to connect to OpenAI realtime: {e}").send()
    #     return False

@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.InputAudioChunk):
    openai_realtime: RealtimeClient = cl.user_session.get("openai_realtime")
    if openai_realtime:            
        if openai_realtime.is_connected():
            # Set flag to indicate this is audio input, not text
            cl.user_session.set("is_text_input", False)
            await openai_realtime.append_input_audio(chunk.data)
        else:
            logger.info("RealtimeClient is not connected")

@cl.on_audio_end
async def on_audio_end():
    return True
@cl.on_chat_end
@cl.on_stop
async def on_end():
    openai_realtime: RealtimeClient = cl.user_session.get("openai_realtime")
    if openai_realtime and openai_realtime.is_connected():
        await openai_realtime.disconnect()