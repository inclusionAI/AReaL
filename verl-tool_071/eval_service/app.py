import os
import json
import argparse
import inspect
import traceback
import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from transformers import HfArgumentParser
import uvicorn
from config import ServerConfig, ModelConfig, ToolConfig
from model_service import ModelService

# Set up logging
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("error_log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_app(server_config: ServerConfig, model_config: ModelConfig, tool_config: ToolConfig) -> FastAPI:
    """
    Create and configure the FastAPI application
    
    Args:
        server_config: Server configuration object
        model_config: Model configuration object
        tool_config: Tool configuration object
        
    Returns:
        Configured FastAPI application instance
    """
    app = FastAPI(
        title="LLM Code Tool Service",
        description="Large language model code tool calling service compatible with OpenAI API",
        version="1.0.0"
    )
    
    # Add CORS middleware to allow cross-origin requests
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Set debug mode based on environment
    if hasattr(server_config, "environment") and server_config.environment == "development":
        app.debug = True
    
    # Initialize the model service
    model_service = ModelService(model_config, tool_config)
    model_service.load_model()
    
    # Store service in application state
    app.state.model_service = model_service
    
    # Add middleware for global exception handling
    @app.middleware("http")
    async def log_exceptions(request: Request, call_next):
        try:
            return await call_next(request)
        except Exception as e:
            error_details = traceback.format_exc()
            logger.error(f"Unhandled exception: {str(e)}\n{error_details}")
            raise
    
    @app.post("/completions")
    async def chat_completions(request: Request):
        """
        Chat completion API endpoint compatible with OpenAI
        
        Processes chat messages and returns model-generated responses with tool calling capabilities
        """
        try:
            request_body = await request.json()
            logger.debug(f"Received completions request: {json.dumps(request_body)}")
            response = await app.state.model_service.completions_async(request_body)
            return response
        except Exception as e:
            error_details = traceback.format_exc()
            logger.error(f"Error in completions endpoint: {str(e)}\n{error_details}")
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    @app.post("/chat/completions")
    async def completions(request: Request):
        """
        Chat completion API endpoint compatible with OpenAI
        
        Processes chat messages and returns model-generated responses with tool calling capabilities
        """
        try:
            request_body = await request.json()
            logger.debug(f"Received chat completions request: {json.dumps(request_body)}")
            response = await app.state.model_service.chat_completions_async(request_body)
            return response
        except Exception as e:
            error_details = traceback.format_exc()
            logger.error(f"Error in chat completions endpoint: {str(e)}\n{error_details}")
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint to verify service availability"""
        return {"status": "healthy"}
    
    return app

async def main_async():
    # Set up command line argument parsing
    hf_parser = HfArgumentParser((ServerConfig, ModelConfig, ToolConfig))
    server_config, model_config, tool_config = hf_parser.parse_args_into_dataclasses()    
    tool_config.post_init()
    
    # Create and run the application
    app = create_app(server_config, model_config, tool_config)
    
    # Configure and start the server with enhanced logging
    config = uvicorn.Config(
        app, 
        host=server_config.host, 
        port=server_config.port, 
        log_level=server_config.log_level,  # Changed from "error" to "debug" for better visibility
        ws_max_queue=server_config.ws_max_queue, 
        workers=server_config.workers*model_config.num_models,
        access_log=True,
        timeout_keep_alive=server_config.timeout_keep_alive  # Added keep-alive timeout setting
    )
    server = uvicorn.Server(config)
    await server.serve()

def main():
    import asyncio
    asyncio.run(main_async())

if __name__ == "__main__":
    main()