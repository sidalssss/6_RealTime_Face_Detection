import cv2
import time
import json
import logging
import threading
from typing import List, Tuple
from datetime import datetime

# Enterprise Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FaceAnalytics")

class AdvancedFaceDetector:
    """
    Çoklu iş parçacığı (Threading) destekli gerçek zamanlı yüz tespiti ve analizi sistemi.
    Haar Cascade ve Gelişmiş Filtreleme teknikleri kullanılarak optimize edilmiştir.
    """
    def __init__(self, cascade_path: str = None):
        if cascade_path is None:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.is_running = False
        self.session_data = []
        self._lock = threading.Lock()

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Tuple]]:
        """Kareyi gri tonlamaya çevirir, histogram eşitleme uygular ve yüzleri tespit eder."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.15, minNeighbors=8, minSize=(40, 40)
        )
        return frame, faces

    def start_engine(self):
        """Kamera akışını başlatır ve analiz motorunu çalıştırır."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Hata: Kamera kaynağına erişilemedi.")
            return

        self.is_running = True
        prev_time = 0
        logger.info("Sidal AI Analytics Engine v3.0 Başlatıldı.")

        while self.is_running:
            ret, frame = cap.read()
            if not ret: break

            # FPS Analizi
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time

            # Görüntü İşleme
            frame, faces = self.process_frame(frame)

            # Görselleştirme ve Veri Kaydı
            with self._lock:
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 127), 2)
                    cv2.putText(frame, "Sidal-AI: FaceDetected", (x, y-10), 
                                cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 127), 1)
                    
                    self.session_data.append({
                        "timestamp": datetime.now().isoformat(),
                        "bbox": [int(x), int(y), int(w), int(h)],
                        "confidence": "High"
                    })

            # Telemetri Bilgisi
            cv2.putText(frame, f"FPS: {int(fps)} | Detections: {len(faces)}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow('Sidal AI - Enterprise Face Analytics', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.is_running = False

        self.save_analytics()
        cap.release()
        cv2.destroyAllWindows()

    def save_analytics(self, filename: str = "analytics_log.json"):
        """Oturum verilerini kalıcı depolamaya aktarır."""
        with open(filename, "w") as f:
            json.dump(self.session_data, f, indent=4)
        logger.info(f"Oturum Analizi Kaydedildi: {filename} ({len(self.session_data)} kayıt)")

if __name__ == "__main__":
    detector = AdvancedFaceDetector()
    detector.start_engine()
