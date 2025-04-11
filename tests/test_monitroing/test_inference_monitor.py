"""Tests für InferenceMonitor-Komponente."""

import json
import os
import shutil
import tempfile
import time

import pytest

from tinylcm.core.monitoring.inference_monitor import InferenceMonitor
from tinylcm.core.monitoring.anomaly_detectors import ThresholdAnomalyDetector


class TestInferenceMonitor:
    """Test InferenceMonitor-Funktionalität."""
    
    def setup_method(self):
        """Temporäres Verzeichnis für Tests einrichten."""
        self.temp_dir = tempfile.mkdtemp()
        self.monitor = InferenceMonitor(
            storage_dir=self.temp_dir,
            log_interval=5  # Kleineres Intervall für Tests
        )
    
    def teardown_method(self):
        """Ressourcen und temporäres Verzeichnis aufräumen."""
        self.monitor.close()
        shutil.rmtree(self.temp_dir)
    
    def test_track_inference(self):
        """Test, dass eine einzelne Inferenz verfolgt wird."""
        # Eine Inferenz verfolgen
        record = self.monitor.track_inference(
            input_id="test_input_1",
            prediction="cat",
            confidence=0.85,
            latency_ms=15.3,
            ground_truth="cat"
        )
        
        # Aufzeichnung prüfen
        assert record["input_id"] == "test_input_1"
        assert record["prediction"] == "cat"
        assert record["confidence"] == 0.85
        assert record["latency_ms"] == 15.3
        assert record["ground_truth"] == "cat"
        assert "timestamp" in record
        assert "session_id" in record
        
        # Prüfe, dass Metriken aktualisiert wurden
        metrics = self.monitor.get_current_metrics()
        assert metrics["total_inferences"] == 1
        assert "cat" in metrics["prediction_distribution"]
    
    def test_get_current_metrics(self):
        """Test, dass aktuelle Überwachungsmetriken zurückgegeben werden."""
        # Füge einige Inferenzen hinzu
        for i in range(5):
            self.monitor.track_inference(
                input_id=f"input_{i}",
                prediction="cat" if i % 2 == 0 else "dog",
                confidence=0.8 + i * 0.02,
                latency_ms=10.0 + i * 2
            )
        
        # Hole Metriken
        metrics = self.monitor.get_current_metrics()
        
        # Prüfe Metriken
        assert metrics["total_inferences"] == 5
        assert "session_id" in metrics
        assert "prediction_distribution" in metrics
        assert metrics["prediction_distribution"]["cat"] == 3
        assert metrics["prediction_distribution"]["dog"] == 2
        
        # Prüfe Latenzstatistiken
        assert "latency" in metrics
        assert "mean_ms" in metrics["latency"]
        assert "median_ms" in metrics["latency"]
        assert "min_ms" in metrics["latency"]
        assert "max_ms" in metrics["latency"]
        assert "p95_ms" in metrics["latency"]
        
        # Prüfe Konfidenzstatistiken
        assert "confidence" in metrics
        assert "mean" in metrics["confidence"]
        assert "median" in metrics["confidence"]
        assert "min" in metrics["confidence"]
        assert "max" in metrics["confidence"]
    
    def test_register_anomaly_callback(self):
        """Test, dass Anomalie-Callbacks registriert und ausgelöst werden."""
        # Erstelle einen Test-Callback
        callback_calls = []
        
        def anomaly_callback(record):
            callback_calls.append(record)
        
        # Registriere den Callback
        self.monitor.register_anomaly_callback(anomaly_callback)
        
        # Erstelle einen speziellen Anomaliedetektor mit niedrigem Grenzwert
        low_threshold_detector = ThresholdAnomalyDetector(confidence_threshold=0.8)
        self.monitor.anomaly_detector = low_threshold_detector
        
        # Verfolge eine normale Inferenz (sollte keine Anomalie sein)
        self.monitor.track_inference(
            input_id="normal",
            prediction="cat",
            confidence=0.9,
            latency_ms=10.0
        )
        
        # Keine Anomalien bisher
        assert len(callback_calls) == 0
        
        # Verfolge eine Inferenz mit niedriger Konfidenz (sollte Anomalie auslösen)
        self.monitor.track_inference(
            input_id="low_conf",
            prediction="cat",
            confidence=0.7,  # Unter dem Grenzwert
            latency_ms=10.0
        )
        
        # Callback sollte aufgerufen worden sein
        assert len(callback_calls) == 1
        assert callback_calls[0]["input_id"] == "low_conf"
        assert callback_calls[0]["anomaly"] is True
        assert "anomaly_reasons" in callback_calls[0]
        assert any("confidence" in reason.lower() for reason in callback_calls[0]["anomaly_reasons"])
    
    def test_export_metrics(self):
        """Test, dass Metriken in eine Datei exportiert werden."""
        # Füge einige Inferenzen hinzu
        for i in range(10):
            self.monitor.track_inference(
                input_id=f"input_{i}",
                prediction="class_" + str(i % 3),
                confidence=0.7 + (i % 3) * 0.1,
                latency_ms=10.0 + i
            )
        
        # Exportiere als JSON
        json_path = self.monitor.export_metrics(format="json")
        
        # Prüfe, dass Datei existiert
        assert os.path.exists(json_path)
        
        # Prüfe Inhalt
        with open(json_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        
        assert metrics["total_inferences"] == 10
        assert "prediction_distribution" in metrics
        assert "latency" in metrics
        assert "confidence" in metrics
        
        # Exportiere als CSV
        csv_path = self.monitor.export_metrics(format="csv")
        
        # Prüfe, dass Datei existiert
        assert os.path.exists(csv_path)
        
        # Prüfe Inhalt (grundlegende Prüfung)
        with open(csv_path, "r", encoding="utf-8") as f:
            csv_content = f.read()
        
        assert "total_inferences" in csv_content
        assert "10" in csv_content  # Sollte die Anzahl enthalten
    
    def test_log_file_creation(self):
        """Test, dass Protokolldateien erstellt werden."""
        # Füge genau log_interval Inferenzen hinzu
        for i in range(self.monitor.log_interval):
            self.monitor.track_inference(
                input_id=f"input_{i}",
                prediction="cat",
                confidence=0.9,
                latency_ms=10.0
            )
        
        # Prüfe, dass eine Protokolldatei erstellt wurde
        log_files = [f for f in os.listdir(self.temp_dir) if f.startswith("inference_log_")]
        assert len(log_files) == 1
        
        # Lese die Datei und prüfe den Inhalt
        log_path = os.path.join(self.temp_dir, log_files[0])
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        assert len(lines) == self.monitor.log_interval
        
        # Prüfe, dass jede Zeile gültiges JSON ist
        for line in lines:
            record = json.loads(line)
            assert "input_id" in record
            assert "prediction" in record
    
    def test_context_manager(self):
        """Test, dass InferenceMonitor als Context Manager verwendet werden kann."""
        # Erstelle einen Monitor mit Context Manager
        with InferenceMonitor(storage_dir=self.temp_dir) as monitor:
            # Füge eine Inferenz hinzu
            monitor.track_inference(
                input_id="test_cm",
                prediction="cat",
                confidence=0.9,
                latency_ms=10.0
            )
            
            # Prüfe, dass Metriken verfügbar sind
            metrics = monitor.get_current_metrics()
            assert metrics["total_inferences"] == 1
        
        # Nach dem Context Manager sollte close() aufgerufen worden sein
        # Prüfe, dass Metrikdateien erstellt wurden
        metrics_files = [f for f in os.listdir(self.temp_dir) if f.startswith("metrics_")]
        assert len(metrics_files) > 0