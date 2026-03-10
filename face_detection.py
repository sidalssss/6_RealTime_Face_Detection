import cv2
import time
import json
import threading
from datetime import datetime

class AdvancedFaceDetector:
    """Gerçek zamanlı yüz tespiti ve analizi yapan profesyonel modül."""
    def __init__(self, cascade_path=None):
        if cascade_path is None:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.is_running = False
        self.detections_log = []

    def start_streaming(self):
        """Kameradan akışı başlatır ve analiz eder."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[HATA] Kamera açılamadı.")
            return

        self.is_running = True
        prev_time = 0
        
        print("\nSidal AI - Real-Time Face Analytics Başlatıldı...")
        print("Çıkmak için 'q' tuşuna basın, kayıt almak için 's' tuşuna basın.")

        while self.is_running:
            ret, frame = cap.read()
            if not ret: break

            # 1. FPS Hesaplama
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            # 2. Gri Tonlama ve İşleme
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Histogram eşitleme (Düşük ışıkta daha iyi sonuç için)
            gray = cv2.equalizeHist(gray)
            
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.2, 
                minNeighbors=7, 
                minSize=(30, 30)
            )

            # 3. Görselleştirme
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # Profesyonel etiket
                cv2.putText(frame, f"Person: FaceDetected (Conf: High)", (x, y-10), 
                            cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 1)
                
                # Koordinatları kaydet
                self.detections_log.append({
                    "timestamp": datetime.now().isoformat(),
                    "coords": [int(x), int(y), int(w), int(h)]
                })

            # 4. Bilgi Panelini Göster
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f"Detections: {len(faces)}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            cv2.imshow('Sidal AI Analytics Engine', frame)

            # 5. Klavye Kontrolleri
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.is_running = False
            elif key == ord('s'):
                self.save_log()

        cap.release()
        cv2.destroyAllWindows()

    def save_log(self):
        """Tespit verilerini JSON dosyasına kaydeder."""
        with open("detections_history.json", "w") as f:
            json.dump(self.detections_log, f, indent=4)
        print(f"\n[BİLGİ] {len(self.detections_log)} tespit verisi kaydedildi.")

if __name__ == "__main__":
    detector = AdvancedFaceDetector()
    detector.start_streaming()
