import cv2
import time

def main():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

    gender_net = cv2.dnn.readNetFromCaffe(
        "deploy_gender.prototxt",
        "gender_net.caffemodel"
    )
    gender_list = ['Masculino', 'Feminino']

    observed_faces = {}

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erro: Não foi possível acessar a câmera.")
        return

    print("Pressione 'q' para sair.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro: Não foi possível capturar o frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for i, (x, y, w, h) in enumerate(faces):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            face_roi = frame[y:y + h, x:x + w]

            blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
            gender_net.setInput(blob)
            gender_pred = gender_net.forward()
            gender = gender_list[gender_pred[0].argmax()]

            face_id = f"{x}_{y}_{w}_{h}"

            if face_id not in observed_faces:
                observed_faces[face_id] = time.time()
            elapsed_time = time.time() - observed_faces[face_id]

            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=20, minSize=(25, 25))

            is_smiling = "Sim" if len(smiles) > 0 else "Não"

            text_offset = 10
            text_x = x + w + text_offset
            text_y = y

            cv2.rectangle(frame, (text_x - 5, text_y - 20), (text_x + 200, text_y + 100), (0, 0, 0), -1)

            cv2.putText(frame, "Nome: Hayden Fernandes de Andrade", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "Idade: 34 anos", (text_x, text_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Gênero: {gender}", (text_x, text_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Tempo: {elapsed_time:.1f} seg", (text_x, text_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Sorrindo: {is_smiling}", (text_x, text_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

       
        cv2.imshow("Projeto para detectar rosto e emoções básicas", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
