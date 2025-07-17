import os
import chainlit as cl

# Set up the locales for internationalization support
cl.configure(
    default_language="en",
    translations_dir=os.path.join(os.path.dirname(__file__), "locales"),
    supported_languages=["en", "ko"]
)
