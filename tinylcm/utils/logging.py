"""Logging utilities for TinyLCM.

Provides consistent and configurable logging capabilities for all TinyLCM components,
with support for console and file logging, formatting, and level control.
"""

import logging
import os
import sys
from typing import Optional

def setup_logger(
    name: str, 
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Konfiguriert und gibt einen Logger zurück.
    
    Args:
        name: Name des Loggers
        level: Logging-Level (default: INFO)
        log_file: Optionaler Pfad zur Log-Datei
        log_format: Optionales Format für Log-Nachrichten
        
    Returns:
        Konfigurierter Logger
    """
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Entferne bestehende Handler, um Duplizierung zu vermeiden
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Erstelle einen Handler für die Konsole
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)
    
    # Füge optional einen Datei-Handler hinzu
    if log_file:
        # Stelle sicher, dass das Verzeichnis existiert
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)
    
    return logger